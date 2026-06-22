# 深度学习与空间智能期末作业工程

本项目对应“基于 3DGS/AIGC 的多源资产融合与 ACT 跨环境泛化实验”。工程已按可复现方式组织脚本、输入、日志、表格和报告；所有已写入报告的状态都来自真实运行日志或真实输出，没有补造训练指标。

## 目录结构

- `inputs/`: 学生上传的物体 A 多视角照片、物体 C 杯子照片。
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

物体 A 原始照片已放入：

```bash
inputs/object_A/
```

物体 C 杯子照片已放入：

```bash
inputs/object_C/cup.jpg
```

重新预处理：

```bash
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/extract_frames.py
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/preprocess_object_c.py
```

物体 C 的去背景结果是启发式 RGBA，不是 rembg/SAM 级别的精细 matting，输出见 `outputs/task1/figures/object_C_bg_removal.png`。

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

注意：本次 COLMAP 两轮 mapper 和 patched official convert 均失败，原因是 `No good initial image pair found`，因此没有合法 sparse model，3DGS 训练没有启动。

物体 B 文本到 3D：

```bash
bash scripts/run_threestudio_object_B.sh
```

物体 C 单图到 3D：

```bash
bash scripts/run_zero123_object_C.sh
```

背景 3DGS：

```bash
SCENE=counter bash scripts/train_3dgs_background.sh
```

LeRobot ACT：

```bash
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/prepare_calvin_splits.py
bash scripts/train_act_A.sh
bash scripts/train_act_ABC.sh
bash scripts/eval_act_D.sh
```

本次 Hugging Face 数据探测被代理/SSL 问题阻塞，见 `logs/calvin_hf_metadata_probe_*.log`。脚本没有按 episode index 臆造 A/B/C/D 切分。

## 结果复现

已实际运行并生成输出：

```bash
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/make_report_assets.py
/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/collect_results.py
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=report report/main.tex
```

关键输出：

- 报告：`report/report.pdf`
- 物体 C RGBA：`outputs/task1/object_C_image3d/cup_rgba.png`
- 背景去除对比图：`outputs/task1/figures/object_C_bg_removal.png`
- 结果表：`report/tables/task1_assets.csv`
- CALVIN 探测表：`report/tables/calvin_split_summary.csv`
- 权重占位压缩包：`outputs/weights/hw3_weights.zip`
- 最终检查：`final_checklist.md`

## 常见问题

- CUDA OOM：降低 `ITERATIONS`、`RESOLUTION="-r 8"`，或换空闲 GPU。
- COLMAP 失败：增加物体 A 环绕照片到 80-180 张，保证相邻视角重叠和足够基线；当前 6 张照片不足以初始化稳定几何。
- threestudio 依赖重：建议单独创建 `requirements/env_threestudio.yml` 环境，避免污染 base。
- CALVIN/Hugging Face 失败：检查服务器代理和 SSL，确认能访问 `https://huggingface.co/datasets/xiaoma26/calvin-lerobot` 后再运行 split 脚本。

## GitHub 与权重

当前未创建远程 GitHub 仓库。可在确认报告和个人信息后运行：

```bash
git init
git add README.md run_manifest.md requirements configs scripts report final_checklist.md
git commit -m "Add spatial intelligence final project"
git remote add origin <your-public-repo-url>
git push -u origin main
```

不要把大数据集、模型权重或本地 conda 依赖目录上传到 GitHub。`outputs/weights/hw3_weights.zip` 当前只包含无训练权重说明，需要你后续上传网盘并替换报告链接。
