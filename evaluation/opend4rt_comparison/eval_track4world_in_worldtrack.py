#!/usr/bin/env python3
"""Score Track4World on WorldTrack using Open-d4rt's protocol.

Everything that defines the benchmark is imported from eval_track3d_in_worldtrack
rather than reimplemented, so the numbers land on the same scale as the table in
Open-d4rt's README: same clips, same frame-0 visible queries, same GT anchored to
frame 0, same global-median-scale APD/EPE. Only the predictor is swapped.

Track4World is fed through its own preprocessing (the resolution its
TrackingEvalDataset uses for this benchmark), because input resolution is a
property of the model, not of the protocol. Forcing it into D4RT's 256x256 would
measure the resize, not the tracker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

# Open-d4rt lives beside this repo as a git submodule (third_party/Open-d4rt).
# Set OPEND4RT_ROOT to point elsewhere if you keep it somewhere else.
_REPO_ROOT = Path(__file__).resolve().parents[2]
OPEND4RT_ROOT = Path(os.environ.get("OPEND4RT_ROOT", _REPO_ROOT / "third_party" / "Open-d4rt"))
if not (OPEND4RT_ROOT / "eval_track3d_in_worldtrack.py").exists():
    raise SystemExit(
        f"Open-d4rt not found at {OPEND4RT_ROOT}. Initialise the submodule with\n"
        "  git submodule update --init third_party/Open-d4rt\n"
        "or set OPEND4RT_ROOT to your checkout."
    )
if str(OPEND4RT_ROOT) not in sys.path:
    sys.path.insert(0, str(OPEND4RT_ROOT))

# Import the protocol before Track4World's root goes on sys.path: both repos ship
# top-level modules called utils/model, and the first one imported wins.
from eval_track3d_in_worldtrack import (  # noqa: E402
    _aggregate_results,
    _format_subset_summary,
    _metrics_for_sequence,
    load_worldtrack_sequence,
)

# Track4World's TrackingEvalDataset resolution per subset. ADT is left at its
# native 512x512; everything else goes to 360x640, which is the aspect all three
# of the remaining subsets already have.
SUBSET_MODEL_HW: dict[str, tuple[int, int]] = {
    "adt_mini": (512, 512),
    "ds_mini": (360, 640),
    "po_mini": (360, 640),
    "pstudio_mini": (360, 640),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Track4World with Open-d4rt's WorldTrack protocol.")
    parser.add_argument("--t4w-root", default=str(_REPO_ROOT))
    parser.add_argument("--ckpt-init", default="checkpoints/track4world_da3.pth")
    parser.add_argument("--config-path", default="track4world/config/eval/v1.json")
    parser.add_argument("--coordinate", default="world_depthanythingv3",
                        choices=("camera_base", "world_pi3", "world_depthanythingv3"))
    parser.add_argument("--metric-scale", action="store_true")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "evaluation" / "track"))
    parser.add_argument("--subsets", default="adt_mini,po_mini,pstudio_mini,ds_mini")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--limit-seqs", type=int, default=0)
    # Track4World runs at its own evaluation resolution by default, since input
    # resolution is a property of the model. Override it as "H,W" to ablate that
    # choice, e.g. --model-hw 256,256 to match OpenD4RT's input size.
    parser.add_argument("--model-hw", default="")
    # Equalise image information against a model evaluated at a smaller size: the
    # clip is downsampled to this and upsampled back to --model-hw before inference.
    parser.add_argument("--bottleneck-hw", default="")
    parser.add_argument("--output-dir", default="tmp/eval_worldtrack_t4w")
    parser.add_argument("--save-per-sequence", action="store_true")
    return parser.parse_args()


def build_track4world(args: argparse.Namespace) -> torch.nn.Module:
    root = Path(args.t4w_root).resolve()
    if str(root) not in sys.path:
        sys.path.append(str(root))
    from demo import load_model  # noqa: E402

    config_path = root / args.config_path if not Path(args.config_path).is_absolute() else Path(args.config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    ckpt = root / args.ckpt_init if not Path(args.ckpt_init).is_absolute() else Path(args.ckpt_init)
    if not ckpt.exists():
        raise FileNotFoundError(f"Track4World checkpoint not found: {ckpt}")

    model_args = SimpleNamespace(
        coordinate=args.coordinate,
        ckpt_init=str(ckpt),
        use_original_backbone=False,
        metric_scale=bool(args.metric_scale),
    )
    return load_model(model_args, config)


def _to_frame0(points_cam: torch.Tensor, camera_poses: torch.Tensor) -> torch.Tensor:
    """Map per-frame camera-space points into the frame-0 camera frame.

    points_cam: (T, N, 3) with entry t expressed in camera t.
    camera_poses: (T, 4, 4) camera-to-world.
    Mirrors transform_to_first_frame in Track4World's evaluation/track/eval.py.
    """
    poses = camera_poses.to(torch.float64)
    transforms = torch.linalg.inv(poses[0]).unsqueeze(0) @ poses
    homo = torch.cat([points_cam.to(torch.float64), torch.ones_like(points_cam[..., :1], dtype=torch.float64)], dim=-1)
    return torch.einsum("tij,tnj->tni", transforms, homo)[..., :3]


def _normalize_poses(camera_poses: torch.Tensor, num_frames: int) -> torch.Tensor:
    poses = camera_poses
    while poses.ndim > 3:
        if poses.shape[0] != 1:
            raise RuntimeError(f"Unexpected camera_poses shape {tuple(camera_poses.shape)}")
        poses = poses[0]
    if poses.shape != (num_frames, 4, 4):
        raise RuntimeError(f"Expected camera_poses (T,4,4) with T={num_frames}, got {tuple(poses.shape)}")
    return poses


@torch.no_grad()
def predict_tracks_ref0(
    *,
    model: torch.nn.Module,
    video_rgb: np.ndarray,
    query_uv_norm: np.ndarray,
    model_hw: tuple[int, int],
    bottleneck_hw: tuple[int, int] | None,
    iters: int,
) -> np.ndarray:
    """Track4World's answer to the protocol's question: (N, T, 3) in the frame-0 frame."""
    num_frames = int(video_rgb.shape[0])
    model_h, model_w = int(model_hw[0]), int(model_hw[1])

    if bottleneck_hw is None:
        frames = np.stack(
            [cv2.resize(frame, (model_w, model_h), interpolation=cv2.INTER_AREA) for frame in video_rgb],
            axis=0,
        )
    else:
        # Route the clip through a smaller canvas and back up, so the model sees no
        # more image information than a model evaluated at bottleneck_hw would,
        # while still running on the canvas it expects.
        bh, bw = int(bottleneck_hw[0]), int(bottleneck_hw[1])
        frames = np.stack(
            [
                cv2.resize(
                    cv2.resize(frame, (bw, bh), interpolation=cv2.INTER_AREA),
                    (model_w, model_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                for frame in video_rgb
            ],
            axis=0,
        )
    rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float().cuda()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output, _ = model.infer(
            rgbs,
            iters=int(iters),
            sw=None,
            is_training=False,
            tracking3d=True,
            force_projection=True,
            use_da3_focal=False,
            eval_dict=None,
        )

    flow_3d = output[1]["flow_3d"]  # (1, T, H, W, 3), entry t in camera t
    if flow_3d.ndim != 5 or flow_3d.shape[0] != 1:
        raise RuntimeError(f"Unexpected flow_3d shape {tuple(flow_3d.shape)}")
    poses = _normalize_poses(output[0]["camera_poses"], num_frames)

    # The dense maps are indexed by the query's pixel in the window's first frame,
    # and every query here is a frame-0 query, so a single lookup is enough.
    px = np.rint(np.asarray(query_uv_norm, dtype=np.float64)[:, 0] * (model_w - 1)).astype(np.int64)
    py = np.rint(np.asarray(query_uv_norm, dtype=np.float64)[:, 1] * (model_h - 1)).astype(np.int64)
    px = np.clip(px, 0, model_w - 1)
    py = np.clip(py, 0, model_h - 1)
    xx = torch.from_numpy(px).cuda()
    yy = torch.from_numpy(py).cuda()

    tracks_cam = flow_3d[0, :, yy, xx]  # (T, N, 3)
    tracks_ref0 = _to_frame0(tracks_cam, poses)  # (T, N, 3)
    return tracks_ref0.permute(1, 0, 2).contiguous().cpu().numpy().astype(np.float64)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    model = build_track4world(args)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"WorldTrack root not found: {data_root}")

    subsets = [item.strip() for item in str(args.subsets).split(",") if item.strip()]
    all_summary: dict[str, Any] = {
        "inputs": {
            "predictor": "Track4World",
            "ckpt_init": str(args.ckpt_init),
            "coordinate": args.coordinate,
            "iters": int(args.iters),
            "data_root": str(data_root),
            "subsets": subsets,
            "num_frames": int(args.num_frames),
            "protocol": "Open-d4rt eval_track3d_in_worldtrack (frame-0 queries, global median scale)",
        },
        "subsets": {},
    }

    for subset in subsets:
        subset_dir = data_root / subset
        if not subset_dir.exists():
            print(f"[warn] skipping missing subset: {subset_dir}", flush=True)
            continue
        seq_paths = sorted(subset_dir.glob("*.npz"))
        if int(args.limit_seqs) > 0:
            seq_paths = seq_paths[: int(args.limit_seqs)]
        if not seq_paths:
            print(f"[warn] no sequences in {subset_dir}", flush=True)
            continue

        if args.model_hw:
            parts = [int(v) for v in str(args.model_hw).replace("x", ",").split(",")]
            if len(parts) != 2:
                raise ValueError(f"--model-hw wants 'H,W', got {args.model_hw!r}")
            model_hw = (parts[0], parts[1])
        else:
            model_hw = SUBSET_MODEL_HW.get(subset, (360, 640))
        if args.bottleneck_hw:
            parts = [int(v) for v in str(args.bottleneck_hw).replace("x", ",").split(",")]
            if len(parts) != 2:
                raise ValueError(f"--bottleneck-hw wants 'H,W', got {args.bottleneck_hw!r}")
            bottleneck_hw = (parts[0], parts[1])
        else:
            bottleneck_hw = None
        print(
            f"[eval] subset={subset} sequences={len(seq_paths)} model_hw={model_hw}"
            f" bottleneck_hw={bottleneck_hw}",
            flush=True,
        )
        subset_results: list[dict[str, Any]] = []
        subset_out_dir = output_dir / subset
        subset_out_dir.mkdir(parents=True, exist_ok=True)

        for seq_idx, seq_path in enumerate(seq_paths, 1):
            started = time.time()
            sample = load_worldtrack_sequence(seq_path, num_frames=int(args.num_frames))
            video_rgb = sample["video_rgb"]
            original_h = int(video_rgb.shape[1])
            original_w = int(video_rgb.shape[2])

            # Query selection, verbatim from eval_track3d_in_worldtrack.main.
            visible_mask = np.asarray(sample["visibility"][0], dtype=bool)
            if not np.any(visible_mask):
                print(f"[warn] {seq_path.name}: no visible frame-0 queries", flush=True)
                continue
            query_uv = np.asarray(sample["tracks_uv"][0, visible_mask], dtype=np.float64)
            finite_mask = np.isfinite(query_uv).all(axis=-1)
            depth0 = np.asarray(sample["tracks_xyz_cam"][0, visible_mask, 2], dtype=np.float64)
            finite_mask &= np.isfinite(depth0) & (np.abs(depth0) > 1e-8)
            if not np.any(finite_mask):
                print(f"[warn] {seq_path.name}: no finite frame-0 query UVs", flush=True)
                continue
            query_uv = query_uv[finite_mask]
            gt_tracks_world = np.asarray(sample["tracks_xyz_world"][:, visible_mask], dtype=np.float64)[:, finite_mask]
            query_uv_norm = query_uv.astype(np.float32)
            query_uv_norm[:, 0] /= float(max(original_w - 1, 1))
            query_uv_norm[:, 1] /= float(max(original_h - 1, 1))
            query_uv_norm = np.clip(query_uv_norm, 0.0, 1.0)

            pred_tracks_ref0_nt3 = predict_tracks_ref0(
                model=model,
                video_rgb=video_rgb,
                query_uv_norm=query_uv_norm,
                model_hw=model_hw,
                bottleneck_hw=bottleneck_hw,
                iters=int(args.iters),
            )
            pred_tracks_ref0 = pred_tracks_ref0_nt3.transpose(1, 0, 2)

            metrics = _metrics_for_sequence(
                gt_tracks_world=gt_tracks_world,
                pred_tracks_ref0=pred_tracks_ref0,
                compute_dyn=True,
            )
            metrics.update(
                {
                    "video_name": sample["video_name"],
                    "sequence_path": str(seq_path),
                    "model_image_size": [int(model_hw[0]), int(model_hw[1])],
                    "bottleneck_image_size": list(bottleneck_hw) if bottleneck_hw else None,
                    "original_image_size": [original_h, original_w],
                }
            )
            subset_results.append(metrics)
            if args.save_per_sequence:
                with open(subset_out_dir / f"{sample['video_name']}.json", "w", encoding="utf-8") as handle:
                    json.dump(metrics, handle, indent=2)

            print(
                f"  [{seq_idx}/{len(seq_paths)}] {sample['video_name']}  "
                f"APD={metrics.get('avg_pts_global', float('nan')):.4f}  "
                f"EPE={metrics.get('epe_global', float('nan')):.4f}  "
                f"queries={metrics.get('num_queries', -1)}  {time.time() - started:.1f}s",
                flush=True,
            )
            torch.cuda.empty_cache()

        if not subset_results:
            continue
        summary = _aggregate_results(subset_results)
        all_summary["subsets"][subset] = summary
        print(_format_subset_summary(subset, summary), flush=True)
        with open(subset_out_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "sequences": subset_results}, handle, indent=2)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(all_summary, handle, indent=2)
    print(f"[done] wrote {output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
