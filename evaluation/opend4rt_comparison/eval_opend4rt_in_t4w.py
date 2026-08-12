#!/usr/bin/env python3
"""Score OpenD4RT under Track4World's own 3D-tracking protocol.

The mirror image of eval_track4world_in_worldtrack.py: here the protocol is
Track4World's (its TrackingEvalDataset, its scale+shift alignment, its
TAPVid-3D metrics with the 0.01-2.56 m fixed thresholds and visible-only
scoring) and only the predictor is swapped for OpenD4RT.

Run from the Track4World repo root; the dataset paths in eval.py are relative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

T4W_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(T4W_ROOT))
sys.path.insert(0, str(T4W_ROOT / "evaluation" / "track"))
_DEF_D4RT = T4W_ROOT / "third_party" / "Open-d4rt"
_DEF_CKPT = _DEF_D4RT / "checkpoints" / "OpenD4RT_32CLIP_9Dataset_NoAUG"

# Track4World's protocol pieces, imported rather than reimplemented.
from evaluation.track.eval import TrackingEvalDataset  # noqa: E402
from tapvid3d_metrics import compute_tapvid3d_metrics  # noqa: E402
from track4world.utils.alignment import align_points_scale_xyz_shift  # noqa: E402


def _add_opend4rt_to_path(root: str) -> None:
    if root not in sys.path:
        sys.path.append(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenD4RT under Track4World's tracking protocol.")
    parser.add_argument("--d4rt-root", default=str(_DEF_D4RT))
    parser.add_argument("--model-config", default=str(_DEF_CKPT / "model.yaml"))
    parser.add_argument("--ckpt-path", default=str(_DEF_CKPT / "opend4rt.ckpt"))
    parser.add_argument("--dataset", default="adt", choices=("adt", "ds", "po", "pstudio"))
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--model-hw",
        default="",
        help="Override model.input.image_size as 'H,W'. Defaults to the config value.",
    )
    parser.add_argument("--limit-seqs", type=int, default=0)
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def build_opend4rt(args: argparse.Namespace):
    _add_opend4rt_to_path(args.d4rt_root)
    from src.core import load_checkpoint, load_yaml_config, seed_everything
    from src.model import build_model
    from infer_track_3d import _unwrap_state_dict

    cfg = load_yaml_config(args.model_config)
    seed_everything(int(cfg.get_path("experiment.seed", 42)), deterministic=True)
    model = build_model(cfg["model"]).eval()
    state_dict = _unwrap_state_dict(load_checkpoint(Path(args.ckpt_path), map_location="cpu"))
    if not state_dict:
        raise RuntimeError(f"No model weights in {args.ckpt_path}")
    result = model.load_state_dict(state_dict, strict=False)
    print(f"[d4rt] missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}", flush=True)
    model.cuda().eval()
    image_size = cfg.get_path("model.input.image_size", [256, 256])
    if getattr(args, "model_hw", ""):
        # See the note in Open-d4rt's eval_track3d_in_worldtrack.py: the encoder
        # average-pools its patch grid down to model.encoder.max_tokens, so this
        # raises the input without buying the matching token budget.
        image_size = [int(v) for v in str(args.model_hw).replace("x", ",").split(",")]
        if len(image_size) != 2:
            raise ValueError(f"--model-hw wants 'H,W', got {args.model_hw!r}")
    return model, (int(image_size[0]), int(image_size[1]))


@torch.no_grad()
def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)
    model, model_hw = build_opend4rt(args)

    from infer_track_3d import _infer_tracks, _resize_video

    dataset_paths = {
        "adt": "evaluation/track/adt_mini",
        "ds": "evaluation/track/ds_mini",
        "po": "evaluation/track/po_mini",
        "pstudio": "evaluation/track/pstudio_mini",
    }
    test_dataset = TrackingEvalDataset(
        dataset_name=args.dataset,
        root=dataset_paths[args.dataset],
        num_frames=args.num_frames,
    )
    if int(args.limit_seqs) > 0:
        test_dataset.data_list = test_dataset.data_list[: int(args.limit_seqs)]
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    count_all = 0
    metrics_all = {
        "occlusion_accuracy": 0.0,
        "average_jaccard": 0.0,
        "average_pts_within_thresh": 0.0,
        "average_pts_within_thresh_with_occ": 0.0,
    }

    for test_data_blob in tqdm(test_loader):
        rgbs, trajs_g, tracks_w, tracks_uv, vis_g, intrinsics = [item.cuda() for item in test_data_blob]
        if vis_g.shape[-1] == 0:
            continue
        B, T, H, W, _ = rgbs.shape
        N = vis_g.shape[-1]

        # Every kept point is visible at frame 0 (TrackingEvalDataset filters on
        # vis[0]), so first_positive_inds is all zeros and one pass is enough.
        _, first_positive_inds = torch.max(vis_g, dim=1)

        video_rgb = rgbs[0].cpu().numpy().astype(np.uint8)
        video_model_rgb = _resize_video(video_rgb, image_hw=model_hw)

        query_uv = tracks_uv[0, 0].double().cpu().numpy()  # (N, 2) in dataset pixels
        query_uv_norm = query_uv.astype(np.float32).copy()
        query_uv_norm[:, 0] /= float(max(W - 1, 1))
        query_uv_norm[:, 1] /= float(max(H - 1, 1))
        query_uv_norm = np.clip(query_uv_norm, 0.0, 1.0)

        payload = _infer_tracks(
            model=model,
            video_model_rgb=video_model_rgb,
            native_aspect_ratio=float(W) / float(max(H, 1)),
            query_uv_norm=query_uv_norm,
            query_chunk_size=int(args.query_chunk_size),
        )
        # OpenD4RT already reports in the frame-0 reference frame, which is what
        # transform_to_first_frame produces for Track4World, so no pose composition here.
        pred_tracks3d = torch.from_numpy(
            np.asarray(payload["tracks_xyz_ref0"], dtype=np.float32).transpose(1, 0, 2)
        ).cuda().unsqueeze(0)  # (1, T, N, 3)
        pred_vis = torch.from_numpy(np.asarray(payload["tracks_visibility"], dtype=bool)).cuda()  # (N, T)

        query_points_all = torch.cat(
            [first_positive_inds[:, :, None].float(), tracks_uv[:, 0]], dim=2
        )[..., [0, 2, 1]]

        gt_occluded = (vis_g < 0.5).bool().transpose(1, 2)  # (B, N, T)
        pred_occluded = (~pred_vis).unsqueeze(0)  # (B, N, T)

        gt_tracks3d = tracks_w.double().float()
        # Track4World's own alignment, verbatim: fit on valid_mask[::2], scale + xyz shift.
        valid_mask = gt_occluded.transpose(1, 2)  # (B, T, N)
        if gt_tracks3d[valid_mask][::4].shape[0] != 0:
            align_src = pred_tracks3d[valid_mask][::2]
            align_tgt = gt_tracks3d[valid_mask][::2]
        else:
            align_src = pred_tracks3d.reshape(-1, 3)[::4]
            align_tgt = gt_tracks3d.reshape(-1, 3)[::4]
        align_weights = 1 / torch.ones_like(align_tgt.norm(dim=-1))
        finite = torch.isfinite(align_src).all(dim=-1) & torch.isfinite(align_tgt).all(dim=-1)
        if int(finite.sum()) >= 3:
            align_src, align_tgt, align_weights = align_src[finite], align_tgt[finite], align_weights[finite]
        scale, shift = align_points_scale_xyz_shift(align_src, align_tgt, align_weights, exp=10)
        pred_tracks3d_aligned = pred_tracks3d * scale + shift

        out_metrics, _ = compute_tapvid3d_metrics(
            gt_occluded=gt_occluded.transpose(1, 2).cpu().numpy(),
            gt_tracks=gt_tracks3d.cpu().numpy(),
            pred_occluded=pred_occluded.transpose(1, 2).cpu().numpy(),
            pred_tracks=pred_tracks3d_aligned.cpu().numpy(),
            intrinsics_params=intrinsics[0].cpu().numpy(),
            scaling="median",
            query_points=query_points_all.cpu().numpy(),
            order="b t n",
            use_fixed_metric_threshold=True,
            return_scaled_pred=True,
        )
        count_all += 1
        for key in metrics_all:
            metrics_all[key] += float(np.asarray(out_metrics[key]).mean())

    print(f"\n[OpenD4RT under Track4World protocol] dataset={args.dataset} frames={args.num_frames} seqs={count_all}")
    summary = {k: v / max(count_all, 1) for k, v in metrics_all.items()}
    for key, value in summary.items():
        print(f"{key}: {value}")
    if args.out_json:
        json.dump({"dataset": args.dataset, "num_frames": args.num_frames,
                   "num_sequences": count_all, **summary}, open(args.out_json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
