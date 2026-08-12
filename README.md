<div align="center">
<img src='assets/logo.jpg' style="height:100px"></img>
</div>

<h3 align="center"><strong>Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels</strong></h3>
<p align="center">
  <a href="https://github.com/jiah-cloud">Jiahao Lu</a><sup>1</sup>,</span>
  <a href="https://openreview.net/profile?id=~Jiayi_Xu10">Jiayi Xu</a><sup>1</sup>,</span>
  <a href="https://wbhu.github.io/">Wenbo Hu</a><sup>2†</sup>,</span>
  <a href="https://ruijiezhu94.github.io/ruijiezhu/">Ruijie Zhu</a><sup>2</sup>,</span>
  <a href="https://afterjourney00.github.io/">Chengfeng Zhao</a><sup>1</sup>,</span><br>
  <a href="https://saikit.org/index.html">Sai-Kit Yeung</a><sup>1</sup>,</span>
  <a href="https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en">Ying Shan</a><sup>2</sup>,</span>
  <a href="https://liuyuan-pal.github.io/">Yuan Liu</a><sup>1†</sup>
  <br>
  <sup>1</sup> HKUST </span> 
  <sup>2</sup> ARC Lab, Tencent PCG </span>
  <br>
  <b>ECCV 2026</b>
</p>
<div align="center">
  <a href='https://arxiv.org/abs/2603.02573'><img src='https://img.shields.io/badge/arXiv-2603.02573-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href='assets/arxiv_Track4World.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href='https://jiah-cloud.github.io/Track4World.github.io/'><img src='https://img.shields.io/badge/Project-Page-orange'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href='https://huggingface.co/TencentARC/Track4World'><img src='https://img.shields.io/badge/HuggingFace-Weights-yellow'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
  <a href='https://huggingface.co/TencentARC/Track4World/blob/main/LICENSE.txt'><img src='https://img.shields.io/badge/License-Tencent-green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <br>
  <br>
</div>




---

### 🖼️ Framework

<div align="center">
  <img src="assets/framework.png" width="100%" alt="Track4World Teaser">
</div>

**Track4World** estimates dense 3D scene flow of every pixel between arbitrary frame pairs from a monocular video in a global feedforward manner, enabling efficient and dense 3D tracking of every pixel in the world-centric coordinate system.

---

## ⚙️ Setup and Installation

### 1. Clone the Repository

Clone the repository with submodules to ensure all dependencies are included:

```bash
git clone --recursive https://github.com/TencentARC/Track4World.git
cd Track4World
```

### 2. Environment Setup

We provide an installation script tested with **CUDA 12.1** and **Python 3.11**.

```bash
# Create and activate environment
conda create -n track4world python=3.11
conda activate track4world

# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### 3. Install Third-Party Modules

We utilize several external repositories. Please run the following commands to set them up correctly:

```bash
# Install utils3d
git clone https://github.com/jiah-cloud/utils3d.git 

# Setup Pi3 (Sparse checkout)
git clone --no-checkout https://github.com/yyfz/Pi3.git track4world/nets/external/pi3_repo
cd track4world/nets/external/pi3_repo
git sparse-checkout init
git sparse-checkout set pi3
git checkout main
find . -maxdepth 1 -type f -exec rm -f {} \;
mv pi3 ../pi3
cd ../../../..

# Setup Grounded-SAM-2
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git submodules
cd submodules
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd ..
```

### 4. Download Weights

Download the pre-trained model weights and place them in the `checkpoints/` directory.

```bash
mkdir -p checkpoints

# Download SAM2 weights
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O ./checkpoints/sam2.1_hiera_large.pt

# Download Track4World weights
wget https://huggingface.co/TencentARC/Track4World/resolve/main/track4world_da3.pth -O ./checkpoints/track4world_da3.pth
wget https://huggingface.co/TencentARC/Track4World/resolve/main/track4world_pi3.pth -O ./checkpoints/track4world_pi3.pth
wget https://huggingface.co/TencentARC/Track4World/resolve/main/track4world_moge.pth -O ./checkpoints/track4world_moge.pth
```

* **Manual Download:** [HuggingFace Link](https://huggingface.co/TencentARC/Track4World)

---

## 🚀 Demo

Run the following commands to perform tracking and reconstruction on the provided demo video (`demo_data/cat.mp4`).

### 1. First Frame 3D Tracking (`3d_ff`)

Reconstructs 3D motion based on the geometry of the first frame.

```bash
python demo.py \
    --mp4_path demo_data/cat.mp4 \
    --mode 3d_ff \
    --Ts -1 \
    --save_base_dir results/cat
```

### 2. Dense Tracking: Every Pixel, Every Frame (`3d_efep`)

Performs dense 3D tracking for every pixel across all frames.

**Option A: Camera-Centric Coordinate System**
```bash
python demo.py \
    --mp4_path demo_data/cat.mp4 \
    --coordinate world_depthanythingv3 \
    --mode 3d_efep \
    --Ts -1 \
    --ckpt_init checkpoints/track4world_da3.pth \
    --save_base_dir results/cat
```

**Option B: World-Centric Coordinate System**

For world-centric reconstruction, you can also directly run **Step 2** to obtain world-centric 3D tracking results. However, for better visualization, especially to clearly separate foreground and background objects,it is recommended to first segment dynamic objects using DINO and SAM2 in **Step 1**. You can use either `world_depthanythingv3` or `world_pi3` for world coordinate system.

```bash
# 1. DINO + SAM2 Segmentation
# Use --text-prompt to specify the dynamic objects in your video (e.g., "cat.", "person.", "car.").
python scripts/run_dino_sam2.py \
    --video-path demo_data/cat.mp4 \
    --sam2-checkpoint checkpoints/sam2.1_hiera_large.pt \
    --output-dir results/cat \
    --text-prompt "cat."
    
# 2. Run Track4World 3D EFEP
python demo.py \
    --mp4_path demo_data/cat.mp4 \
    --coordinate world_depthanythingv3 \
    --mode 3d_efep \
    --Ts -1 \
    --ckpt_init checkpoints/track4world_da3.pth \
    --save_base_dir results/cat
```

### 3. 2D Tracking (`2d`)

Performs standard 2D tracking in image space.

```bash
python demo.py \
    --mp4_path demo_data/cat.mp4 \
    --mode 2d \
    --Ts -1 \
    --save_base_dir results/cat
```

### 4. Metric Scale Output

When using the DA3 backbone, you can enable `--metric_scale` to output all geometric results (`points`, `flow3d`, `world_points`, `camera_poses`) in metric (meter-level) coordinates:

```bash
python demo.py \
    --mp4_path demo_data/cat.mp4 \
    --coordinate world_depthanythingv3 \
    --mode 3d_efep \
    --Ts -1 \
    --ckpt_init checkpoints/track4world_da3.pth \
    --metric_scale \
    --save_base_dir results/cat
```

> **Note:** Metric scale recovery is currently only supported with the DA3 backbone (`--use_model depthanythingv3`). MoGe and Pi3 backbones output in relative scale.

---

## ✨ Visualization

Visualize the dense 4D trajectories and reconstructed scenes using the generated output files.

**Visualize First Frame 3D Tracking:**

```bash
python visualization/vis_3d_ff.py --ply_dir results/cat/3d_ff_output
```

**Visualize Dense Tracking (Every Pixel):**

```bash
# Camera Centric Visualization
python visualization/vis_3d_efep.py --ply_dir results/cat/3d_efep_output

# World Centric Visualization (Foreground-Background Separation, Static Background)
python visualization/vis_3d_efep_world.py --ply_dir results/cat/3d_efep_output
```

<div align="center">
  <img src="assets/demo_world.gif" width="100%" alt="Visualization Demo">
</div>

---

## 📊 Evaluation

For detailed instructions on how to evaluate the model on standard benchmarks (Sintel, KITTI, Kubric, etc.), please refer to the evaluation guide:

👉 **[Evaluation Guide (evaluation/eval.md)](evaluation/eval.md)**


### 📈 WorldTrack: comparison with OpenD4RT

Track4World and OpenD4RT are scored on the four WorldTrack subsets (50 clips each)
under **both** projects' evaluation protocols. In each protocol only the predictor is
swapped — the data loader, the alignment and the metric code are the host protocol's
own. Track4World is `track4world_da3.pth` with `--coordinate world_depthanythingv3`;
OpenD4RT is its released `OpenD4RT_32CLIP_9Dataset_NoAUG`.

Both models are fed the **same image information**: every clip is downsampled to the
256x256 OpenD4RT consumes and then upsampled back to Track4World's own canvas. The
bottleneck is bit-identical to OpenD4RT's input (both go through Open-d4rt's
`_resize_video`, i.e. `cv2.INTER_AREA`), so neither model sees more of the image than
the other.

#### How to reproduce

OpenD4RT lives beside this repo as a git submodule, so both directions of the swap run
against its unmodified code:

```bash
git submodule update --init third_party/Open-d4rt
# then follow third_party/Open-d4rt/README.md to fetch OpenD4RT_32CLIP_9Dataset_NoAUG
# into third_party/Open-d4rt/checkpoints/
```

Two adapters under `evaluation/opend4rt_comparison/` each keep the host protocol's data
loader, alignment and metrics and only replace the predictor. `--bottleneck-hw` /
`--bottleneck_hw` route each clip through 256x256 and back up, matching the image
information available to OpenD4RT:

```bash
# Track4World under Open-d4rt's protocol
python evaluation/opend4rt_comparison/eval_track4world_in_worldtrack.py \
    --data-root evaluation/track --num-frames 64 --bottleneck-hw 256,256

# OpenD4RT under Track4World's protocol
python evaluation/opend4rt_comparison/eval_opend4rt_in_t4w.py \
    --dataset adt --num_frames 16
```

#### Results (image information matched)

Track4World's numbers are its info-matched score; OpenD4RT's are unchanged, since it
already runs at 256x256. Each protocol reports its own metrics; best per cell in bold.

Under **Open-d4rt's protocol** (frame-0 visible queries, global median-scale alignment).
**APD** (↑):

| Subset | T4W 16f | D4RT 16f | T4W 50f | D4RT 50f | T4W 64f | D4RT 64f |
|---|---:|---:|---:|---:|---:|---:|
| `adt` | **0.8676** | 0.7534 | **0.8214** | 0.7197 | **0.7971** | 0.6991 |
| `po` | **0.7332** | 0.6880 | **0.6919** | 0.6793 | **0.6667** | 0.6604 |
| `pstudio` | 0.7851 | **0.8142** | 0.7395 | **0.8053** | 0.7286 | **0.7863** |
| `ds` | 0.6996 | **0.7211** | 0.7032 | **0.7284** | 0.6991 | **0.7265** |
| **mean** | **0.7714** | 0.7442 | **0.7390** | 0.7332 | **0.7229** | 0.7181 |

**EPE** (↓):

| Subset | T4W 16f | D4RT 16f | T4W 50f | D4RT 50f | T4W 64f | D4RT 64f |
|---|---:|---:|---:|---:|---:|---:|
| `adt` | **0.1478** | 0.2421 | **0.1862** | 0.2712 | **0.2212** | 0.2966 |
| `po` | **0.2589** | 0.3330 | **0.3041** | 0.3216 | **0.3323** | 0.3397 |
| `pstudio` | 0.1968 | **0.1663** | 0.2200 | **0.1693** | 0.2274 | **0.1812** |
| `ds` | 0.3091 | **0.3023** | 0.3062 | **0.2941** | 0.3088 | **0.2944** |
| **mean** | **0.2282** | 0.2609 | **0.2541** | 0.2640 | **0.2724** | 0.2780 |

Under **Track4World's protocol** (isotropic scale + 3D shift alignment, TAPVid-3D
thresholds). **APD** (↑):

| Subset | T4W 16f | D4RT 16f | T4W 50f | D4RT 50f | T4W 64f | D4RT 64f |
|---|---:|---:|---:|---:|---:|---:|
| `adt` | **0.6283** | 0.5051 | **0.5669** | 0.5089 | **0.5440** | 0.4864 |
| `po` | **0.5386** | 0.5143 | **0.5131** | 0.5026 | **0.5045** | 0.4887 |
| `pstudio` | 0.5914 | **0.5993** | 0.5456 | **0.5853** | 0.5365 | **0.5666** |
| `ds` | 0.4975 | **0.5073** | 0.5095 | **0.5234** | 0.5101 | **0.5234** |
| **mean** | **0.5639** | 0.5315 | **0.5338** | 0.5301 | **0.5238** | 0.5163 |

**AJ** (↑):

| Subset | T4W 16f | D4RT 16f | T4W 50f | D4RT 50f | T4W 64f | D4RT 64f |
|---|---:|---:|---:|---:|---:|---:|
| `adt` | **0.5817** | 0.4533 | **0.5194** | 0.4390 | **0.4965** | 0.4146 |
| `po` | **0.3965** | 0.3732 | **0.3555** | 0.3446 | **0.3466** | 0.3300 |
| `pstudio` | 0.5186 | **0.5233** | 0.4442 | **0.4998** | 0.4244 | **0.4739** |
| `ds` | 0.4389 | **0.4447** | 0.4407 | **0.4453** | 0.4391 | **0.4418** |
| **mean** | **0.4839** | 0.4486 | **0.4400** | 0.4322 | **0.4266** | 0.4151 |

**OA** (↑):

| Subset | T4W 16f | D4RT 16f | T4W 50f | D4RT 50f | T4W 64f | D4RT 64f |
|---|---:|---:|---:|---:|---:|---:|
| `adt` | **0.9878** | 0.9724 | **0.9812** | 0.9350 | **0.9792** | 0.9185 |
| `po` | 0.8004 | **0.8164** | 0.7681 | **0.7969** | 0.7649 | **0.7933** |
| `pstudio` | **0.9367** | 0.9291 | 0.8791 | **0.9102** | 0.8605 | **0.8961** |
| `ds` | **0.9721** | 0.9620 | **0.9464** | 0.9332 | **0.9403** | 0.9260 |
| **mean** | **0.9242** | 0.9200 | 0.8937 | **0.8938** | **0.8862** | 0.8835 |

---

## 📝 Citation

If you find **Track4World** useful for your research or applications, please consider citing our paper:

```bibtex
@article{lu2026track4world,
  title={Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels},
  author={Lu, Jiahao and Xu, Jiayi and Hu, Wenbo and Zhu, Ruijie and Zhao, Chengfeng and Yeung, Sai-Kit and Shan, Ying and Liu, Yuan},
  journal={arXiv preprint arXiv:2603.02573},
  year={2026}
}
```

---

## 🤝 Acknowledgements

Our codebase is built upon [MoGe](https://github.com/microsoft/MoGe), [Alltracker](https://github.com/aharley/alltrackerh), [Pi3](https://github.com/yyfz/Pi3), and [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3). We also gratefully acknowledge [TrackingWorld](https://github.com/IGL-HKUST/TrackingWorld) and [VGGT](https://github.com/facebookresearch/vggt) for their excellent work!
