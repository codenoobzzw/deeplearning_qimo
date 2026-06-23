# 深度学习与空间智能期末作业工程

姓名：张之蔚
学号：25210980169
小组成员：独立完成
GitHub：https://github.com/codenoobzzw/deeplearning_qimo.git
权重网盘：https://pan.baidu.com/s/18O2o9Fr9d0nntlCZjT-jNA?pwd=1111 ，提取码：1111

本工程对应课程期末作业：任务一完成 3DGS/AIGC 多源资产生成与融合，任务二完成 LeRobot ACT 在 CALVIN A/B/C/D 环境上的离线泛化实验。报告、脚本、结果表和权重包都已经放在工程内，核心结论以真实运行日志和输出文件为准。

## 主要产物

- 报告：`report/report.pdf`
- 权重包：`outputs/weights/hw3_weights.zip`
- 物体 A 3DGS：`outputs/task1/object_A_3dgs/point_cloud/iteration_1000/point_cloud.ply`
- Mip-NeRF360 counter 背景 3DGS：`outputs/task1/background_3dgs/counter/point_cloud/iteration_1000/point_cloud.ply`
- official-counter 融合点云：`outputs/task1/merged_scene_counter/point_cloud.ply`
- official-counter 漫游视频：`outputs/task1/videos/merged_scene_counter_walkthrough.mp4`
- 视觉 ACT 结果表：`report/tables/task2_visual_calvin_eval_D.csv`

## 目录结构

- `inputs/`：规范化后的物体 A 视频/照片和物体 C 单图。
- `物体/`：原始拍摄文件备份。
- `configs/`：Gaussian 融合 placement 配置。
- `scripts/`：数据准备、训练、合并、渲染、汇总脚本。
- `report/`：LaTeX 报告、图、表和最终 PDF。
- `outputs/`：实验输出、权重、视频、日志派生结果。
- `requirements/`：环境参考文件。
- `third_party/`、`local_pkgs/`、`data/`：本地第三方仓库、依赖和数据缓存。

## 环境概况

已验证服务器 GPU：NVIDIA RTX A6000，CUDA 12.4。

本次实际使用：

- PG 环境：运行预处理、报告和部分 Python 工具。
- base 环境：运行 3DGS 与 LeRobot ACT，PyTorch 2.6.0+cu124。
- `envs/threestudio310`：运行 threestudio DreamFusion/SDS 与 stable-Zero123 烟测。

threestudio 安装 `tiny-cuda-nn`、`nvdiffrast` 等 CUDA 扩展时不够稳定，工程中补了轻量兼容 shim，保证官方 volume/SDS/Zero123 入口可以完成小步数 smoke run。完整长训练建议安装原版 CUDA 扩展后再运行。

## 任务一复现

### 1. 物体 A：视频抽帧 + COLMAP + 3DGS

```bash
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/extract_frames.py \
  --input inputs/object_A/A_video.mp4 \
  --output data/object_A_video/input \
  --stats outputs/task1/object_A_video_frame_stats.json \
  --fps 6 --blur-quantile 0.03 --similarity-threshold 0.999

ITERATIONS=1000 RESOLUTION="-r 2" bash scripts/train_3dgs_object.sh
```

本次结果：90 张抽帧，COLMAP 注册 63 张；3DGS 1000 step 后得到 19133 个 Gaussian，PSNR/SSIM/LPIPS 为 23.9658/0.8707/0.2323。

### 2. 物体 B：threestudio DreamFusion/SDS

官方 SDS 烟测命令：

```bash
bash scripts/run_threestudio_object_B_smoke.sh
```

实际输出：

- `third_party/threestudio/outputs/dreamfusion-sd/a_small_yellow_rubber_duck_toy@20260623-121907/ckpts/last.ckpt`
- `report/figs/object_B_threestudio_sds_smoke.png`

说明：这里使用 tiny Stable Diffusion 做 1 step smoke run，用来验证 threestudio/SDS 代码路径；最终融合展示使用同提示词生成的鸭子 mesh：`outputs/task1/object_B_text3d/export/mesh.obj`。

### 3. 物体 C：去背景 + stable-Zero123

```bash
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/preprocess_object_c.py
bash scripts/run_zero123_object_C_smoke.sh
```

实际输出：

- `outputs/task1/object_C_image3d/cup_rgba.png`
- `third_party/threestudio/outputs/zero123-sai/32_cup_rgba.png@20260623-123104/ckpts/last.ckpt`
- `report/figs/object_C_zero123_smoke.png`

说明：stable-Zero123 权重已经下载并跑通 1 step smoke run；最终融合展示使用基于单图外观生成的杯子 mesh：`outputs/task1/object_C_image3d/export/mesh.obj`。

### 4. Mip-NeRF360 counter 背景 3DGS

```bash
mkdir -p data/mipnerf360
curl -L --retry 5 --retry-delay 5 -C - \
  -o data/mipnerf360/360_v2.zip \
  https://storage.googleapis.com/gresearch/refraw360/360_v2.zip

unzip -q data/mipnerf360/360_v2.zip 'counter/*' -d data/mipnerf360
```

训练与评估：

```bash
SCENE=counter ITERATIONS=1000 RESOLUTION="-r 8" CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train_3dgs_background.sh
```

本次结果：240 张图，初始化 155767 点，最终 210622 个 Gaussian，PSNR/SSIM/LPIPS 为 24.5242/0.8315/0.1954。

### 5. A/B/C 插入 counter 背景并渲染视频

```bash
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/merge_gaussians.py \
  --config configs/placements_official_counter.yaml \
  --output outputs/task1/merged_scene_counter/point_cloud.ply \
  --workdir outputs/task1/merged_scene_counter/intermediate

/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/render_merged_path.py \
  --input outputs/task1/merged_scene_counter/point_cloud.ply \
  --video-output outputs/task1/videos/merged_scene_counter_walkthrough.mp4 \
  --frames-dir outputs/task1/merged_scene_counter/preview_frames \
  --num-frames 48 --width 960 --height 540 --point-radius 1
```

本次结果：293755 个 Gaussian 合并为 official-counter 场景，视频已生成。

## 任务二复现

真实数据来自 `xiaoma26/calvin-lerobot`，本次下载 splitA/B/C/D 每个 split 前 4 个 parquet episode。

state-only ACT 子集实验：

```bash
MPLCONFIGDIR=/tmp/hw3_cache/mpl \
  /home/zhangzhiwei/miniconda3/bin/conda run -n base python scripts/train_act_calvin_subset.py \
  --episodes-per-split 4 --steps 40 --log-every 10 --chunk-size 8 --stride 4 --batch-size 16
```

视觉 ACT 子集实验：

```bash
MPLCONFIGDIR=/tmp/hw3_cache/mpl \
  /home/zhangzhiwei/miniconda3/bin/conda run -n base python scripts/train_act_calvin_visual_subset.py \
  --episodes-per-split 4 --steps 30 --log-every 10 --batch-size 8 --image-size 64 \
  --dim-model 64 --n-heads 4 --dim-feedforward 256 --n-encoder-layers 2
```

视觉 ACT splitD 离线结果：

| 模型 | D mean L1 | D median L1 | L1 < 0.05 | L1 < 0.10 |
| --- | ---: | ---: | ---: | ---: |
| ACT-A-visual-real-subset | 0.278258 | 0.263635 | 0.000000 | 0.000000 |
| ACT-ABC-visual-real-subset | 0.286543 | 0.235508 | 0.000000 | 0.073529 |

该结果是离线 Action L1，不是 CALVIN simulator rollout 成功率。

## 报告和权重

重新生成图表和权重包：

```bash
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/make_report_assets.py
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/collect_results.py
```

重新编译报告：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=report report/main.tex
cp report/main.pdf report/report.pdf
```

当前 `outputs/weights/hw3_weights.zip` 包含：

- ACT 辅助实验权重；
- state-only ACT 真实 CALVIN 子集权重；
- visual ACT 真实 CALVIN 子集权重；
- Object A 3DGS point cloud；
- Mip-NeRF360 counter background 3DGS point cloud；
- official-counter merged scene point cloud；
- threestudio SDS / stable-Zero123 smoke ckpt。
