# 深度学习与空间智能期末作业工程

本项目对应“基于 3DGS/AIGC 的多源资产融合与 ACT 跨环境泛化实验”。工程已按可复现方式组织脚本、输入、日志、表格和报告；所有已写入报告的状态都来自真实运行日志或真实输出，没有补造训练指标。当前版本包含三条线：3DGS/threestudio/Zero123/LeRobot 的官方入口、在外部模型权重和 Mip-NeRF360 大数据不可用时跑通的 A/B/C/背景替代融合闭环，以及基于 `xiaoma26/calvin-lerobot` 真实 parquet 子集的 LeRobot ACTPolicy 离线实验。

## 目录结构

- `inputs/`: 学生上传的物体 A 多视角照片/视频、物体 C 杯子照片。
- `物体/`: 原始拍摄文件备份，与 `inputs/` 中规范化后的输入文件一一对应。
- `data/`: 预处理后的 COLMAP/3DGS/CALVIN 数据目录。
- `third_party/`: 官方 `gaussian-splatting`、`threestudio`、`lerobot`、`calvin` 仓库。
- `scripts/`: 数据准备、训练入口、Gaussian/Mesh 转换、结果收集脚本。
- `outputs/`: 实验输出、日志派生结果、权重压缩包。
- `report/`: 中文 LaTeX 报告、图、表、参考文献。
- `logs/`: 系统探测、安装、COLMAP、预处理、CLI 探测日志。

## 已验证环境

系统探测记录在 `logs/system_info.txt`。普通沙箱下 `nvidia-smi` 不可见；提权运行可见 3 张 NVIDIA RTX A6000，CUDA 12.4。PG 环境为 Python 3.12.9，适合运行数据/报告脚本，但没有 PyTorch。项目已使用 conda `base` 的 PyTorch 2.6.0+cu124，并把 3DGS/LeRobot 补充依赖安装在：

- `local_pkgs/3dgs`
- `local_pkgs/lerobot`

运行 3DGS/LeRobot 时需要设置对应 `PYTHONPATH`，相关训练脚本已内置。

## 数据准备

物体 A 原始照片和后续补拍视频已放入：

```bash
inputs/object_A/
inputs/object_A/A_video.mp4
```

物体 C 杯子照片已放入：

```bash
inputs/object_C/cup.jpg
```

重新预处理：

```bash
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/extract_frames.py
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/extract_frames.py \
  --input inputs/object_A/A_video.mp4 \
  --output data/object_A_video/input \
  --stats outputs/task1/object_A_video_frame_stats.json \
  --fps 6 --blur-quantile 0.03 --similarity-threshold 0.999
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/preprocess_object_c.py
```

物体 C 的去背景结果是启发式 RGBA，不是 rembg/SAM 级别的精细 matting，输出见 `outputs/task1/figures/object_C_bg_removal.png`。原始拍摄文件同时保留在 `物体/` 目录中，便于老师核对原始输入。

## 第三方仓库与依赖

第三方仓库可复现克隆：

```bash
bash scripts/setup_third_party.sh
```

本次实际 clone 的 commit 已记录在 `run_manifest.md` 和 `logs/setup_third_party_retry_*.log`。3DGS 官方 `convert.py` 针对系统 COLMAP 3.6 做了一个兼容补丁：移除了当前 COLMAP 不支持的 `--Mapper.ba_global_function_tolerance` 参数。

3DGS 本地依赖安装命令已记录在 `run_manifest.md`，核心形式为：

```bash
PIP_CACHE_DIR=/tmp/hw3_pip_cache \
/home/zhangzhiwei/miniconda3/bin/conda run -n base python -m pip install \
  --target local_pkgs/3dgs plyfile \
  ./third_party/gaussian-splatting/submodules/diff-gaussian-rasterization \
  ./third_party/gaussian-splatting/submodules/simple-knn \
  ./third_party/gaussian-splatting/submodules/fused-ssim
```

LeRobot 轻量依赖安装后，`lerobot_train --help` 已可通过源码模块运行，完整 help 保存为 `logs/lerobot_train_help.txt`。

## 训练与测试命令

物体 A 3DGS：

```bash
bash scripts/train_3dgs_object.sh
```

本次 6 张原始照片的 COLMAP baseline 仍失败，原因是 `No good initial image pair found`；补拍的 `A_video.mp4` 解决了视角覆盖问题。默认脚本现在使用 `data/object_A_video`，可通过 `SOURCE_DIR`、`OUTPUT_DIR`、`ITERATIONS` 覆盖。已完成的 sanity run 为 90 张抽帧、63 张注册、19133 个 Gaussian、1000 iterations，测试集指标为 PSNR 23.9658、SSIM 0.8707、LPIPS 0.2323。

物体 B 文本到 3D：

```bash
bash scripts/run_threestudio_object_B.sh
```

如果 threestudio 依赖或扩散模型权重不可用，本工程提供已运行的代理资产生成：

```bash
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/generate_proxy_assets.py
```

物体 C 单图到 3D：

```bash
bash scripts/run_zero123_object_C.sh
```

背景 3DGS：

```bash
SCENE=counter bash scripts/train_3dgs_background.sh
```

完整融合与 CPU fallback 漫游渲染：

```bash
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/mesh_to_gaussians.py \
  --mesh outputs/task1/background_3dgs/counter_proxy/export/mesh.obj \
  --output outputs/task1/background_3dgs/counter_proxy/point_cloud/iteration_0001/point_cloud.ply \
  --sample-points 32000 --opacity 0.65 --seed 11
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/merge_gaussians.py \
  --config configs/placements.yaml \
  --output outputs/task1/merged_scene/point_cloud.ply \
  --workdir outputs/task1/merged_scene/intermediate
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/render_merged_path.py \
  --input outputs/task1/merged_scene/point_cloud.ply \
  --video-output outputs/task1/videos/merged_scene_walkthrough.mp4 \
  --frames-dir outputs/task1/merged_scene/preview_frames \
  --num-frames 72 --width 960 --height 540 --point-radius 2
```

LeRobot ACT：

```bash
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/prepare_calvin_splits.py --fallback-known-splits
bash scripts/train_act_A.sh
bash scripts/train_act_ABC.sh
bash scripts/eval_act_D.sh
```

全量 `xiaoma26/calvin-lerobot` 约 69.9GB，本工程没有下载全量数据。为了补上真实数据实验，已下载 `splitA/splitB/splitC/splitD` 每个 split 前 4 个 parquet episode，并运行 LeRobot 官方 `ACTPolicy` 的离线 action chunking 子集训练：

```bash
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python -c "from pathlib import Path; from huggingface_hub import hf_hub_download; repo='xiaoma26/calvin-lerobot'; out=Path('data/calvin/calvin-lerobot-subset'); splits=['splitA','splitB','splitC','splitD']; meta=['info.json','modality.json','episodes.jsonl','tasks.jsonl']; [hf_hub_download(repo, f'{s}/meta/{m}', repo_type='dataset', local_dir=str(out)) for s in splits for m in meta]; [hf_hub_download(repo, f'{s}/data/chunk-000/episode_{i:06d}.parquet', repo_type='dataset', local_dir=str(out)) for s in splits for i in range(4)]"
MPLCONFIGDIR=/tmp/hw3_mpl XDG_CACHE_HOME=/tmp/hw3_cache \
/home/zhangzhiwei/miniconda3/bin/conda run -n base python scripts/train_act_calvin_subset.py \
  --episodes-per-split 4 --steps 40 --log-every 10 --chunk-size 8 --stride 4 --batch-size 16
```

真实子集实验使用 state/environment_state 输入，未使用图像 CNN，也不是 CALVIN simulator rollout success rate。结果见 `outputs/task2/real_calvin_subset_summary.json`、`report/tables/task2_real_calvin_eval_D.csv`、`report/figs/act_real_calvin_action_l1.png`。

为了补齐作业要求中的训练曲线、D 环境评估和权重包，本工程还提供一个已运行的 ACT-style action chunking 代理实验：

```bash
MPLCONFIGDIR=/tmp/hw3_mpl XDG_CACHE_HOME=/tmp/hw3_cache \
/home/zhangzhiwei/miniconda3/bin/conda run -n base python scripts/train_act_proxy.py \
  --steps 800 --log-every 20
```

该代理实验不是 CALVIN 真实 success rate；它模拟 A/B/C/D 视觉分布偏移，使用同一 chunk policy 和超参数训练 ACT-A 与 ACT-ABC，并在 D 上输出 Action L1 与 success rate。

## 结果复现

已实际运行并生成输出：

```bash
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/make_report_assets.py
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/collect_results.py
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=report report/main.tex
cp report/main.pdf report/report.pdf
```

关键输出：

- 报告：`report/report.pdf`
- 物体 A 3DGS：`outputs/task1/object_A_3dgs/point_cloud/iteration_1000/point_cloud.ply`
- 物体 A 渲染指标：`outputs/task1/object_A_3dgs/results.json`
- 物体 A 渲染预览：`report/figs/object_A_3dgs_render_preview.png`
- 物体 B 代理 mesh：`outputs/task1/object_B_text3d/export/mesh.obj`
- 物体 C 代理 mesh：`outputs/task1/object_C_image3d/export/mesh.obj`
- 物体 C RGBA：`outputs/task1/object_C_image3d/cup_rgba.png`
- 背景代理 Gaussian：`outputs/task1/background_3dgs/counter_proxy/point_cloud/iteration_0001/point_cloud.ply`
- 合并场景 PLY：`outputs/task1/merged_scene/point_cloud.ply`
- 漫游视频：`outputs/task1/videos/merged_scene_walkthrough.mp4`
- ACT 真实子集曲线：`report/figs/act_real_calvin_action_l1.png`
- ACT 真实子集 D 评估：`report/tables/task2_real_calvin_eval_D.csv`
- ACT proxy 曲线：`report/figs/act_action_l1_loss.png`
- ACT proxy D 评估：`report/tables/task2_eval_D.csv`
- 背景去除对比图：`outputs/task1/figures/object_C_bg_removal.png`
- 结果表：`report/tables/task1_assets.csv`
- CALVIN 探测表：`report/tables/calvin_split_summary.csv`
- 权重压缩包：`outputs/weights/hw3_weights.zip`
- 最终检查：`final_checklist.md`

## 常见问题

- CUDA OOM：降低 `ITERATIONS`、`RESOLUTION="-r 8"`，或换空闲 GPU。
- COLMAP 失败：6 张照片不足以初始化稳定几何；优先使用 `A_video.mp4` 或重新拍摄 80-180 张环绕照片，保证相邻视角重叠和足够基线。
- threestudio/Zero123 依赖重：建议单独创建 `requirements/env_threestudio.yml` 环境，避免污染 base；当前报告中的 B/C 完整融合使用明确标注的代理 mesh。
- Mip-NeRF360 数据大：当前背景完整融合使用 counter-like 代理 Gaussian；如能下载 counter/garden/bicycle，可用 `scripts/train_3dgs_background.sh` 替换代理背景。
- CALVIN/Hugging Face：全量数据很大，当前只下载每个 split 的 4 个 episode 做真实子集实验；当前 ACT 结果是离线 Action L1，不冒充 simulator 成功率。

## GitHub 与权重

GitHub 仓库：

```text
https://github.com/codenoobzzw/deeplearning_qimo.git
git@github.com:codenoobzzw/deeplearning_qimo.git
```

本工程已包含源码、脚本、输入文件、日志、报告图表、关键输出和提交检查清单。`third_party/`、`local_pkgs/` 和 `data/` 默认不上传到 GitHub，避免把第三方仓库、本地依赖和中间数据缓存一起塞进仓库。

`outputs/weights/hw3_weights.zip` 当前包含 `act_A_proxy.pt`、`act_ABC_proxy.pt`、`ACT-A-real-subset_real_calvin_subset.pt` 与 `ACT-ABC-real-subset_real_calvin_subset.pt`；物体 A 的 3DGS PLY 结果已保留在 `outputs/task1/object_A_3dgs/`。

模型权重百度网盘分享：

```text
链接：https://pan.baidu.com/s/18O2o9Fr9d0nntlCZjT-jNA?pwd=1111
提取码：1111
```
