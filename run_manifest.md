# Run Manifest

All key commands should be run through `scripts/run_and_log.sh` when possible.
This file records commands that were executed for this project and whether they
completed. Results in the report must be traceable to logs or generated files.

## 2026-06-23 latest completion summary

- Object B official SDS smoke completed through threestudio DreamFusion/SDS:
  - command wrapper: `scripts/run_threestudio_object_B_smoke.sh`
  - output ckpt: `third_party/threestudio/outputs/dreamfusion-sd/a_small_yellow_rubber_duck_toy@20260623-121907/ckpts/last.ckpt`
  - report image: `report/figs/object_B_threestudio_sds_smoke.png`
- Object C official stable-Zero123 smoke completed:
  - command wrapper: `scripts/run_zero123_object_C_smoke.sh`
  - output ckpt: `third_party/threestudio/outputs/zero123-sai/32_cup_rgba.png@20260623-123104/ckpts/last.ckpt`
  - report image: `report/figs/object_C_zero123_smoke.png`
- Mip-NeRF360 `counter` downloaded without the 1080 proxy and trained with official gaussian-splatting:
  - data zip: `data/mipnerf360/360_v2.zip`
  - scene: `data/mipnerf360/counter`
  - train log: `logs/background_counter_3dgs_train_20260623_124712.log`
  - render log: `logs/background_counter_3dgs_render_20260623_124812.log`
  - metrics log: `logs/background_counter_3dgs_metrics_20260623_124859.log`
  - result: PSNR 24.5242, SSIM 0.8315, LPIPS 0.1954
- Official-counter merged scene completed:
  - config: `configs/placements_official_counter.yaml`
  - merged PLY: `outputs/task1/merged_scene_counter/point_cloud.ply`
  - video: `outputs/task1/videos/merged_scene_counter_walkthrough.mp4`
- Real visual CALVIN-LeRobot ACT subset completed:
  - script: `scripts/train_act_calvin_visual_subset.py`
  - summary: `outputs/task2/real_calvin_visual_subset_summary.json`
  - D eval table: `report/tables/task2_visual_calvin_eval_D.csv`
  - weights: `outputs/weights/ACT-A-visual-real-subset_real_calvin_visual_subset.pt`, `outputs/weights/ACT-ABC-visual-real-subset_real_calvin_visual_subset.pt`
- Submission artifacts refreshed:
  - report: `report/report.pdf`
  - weights: `outputs/weights/hw3_weights.zip`
  - checklist: `final_checklist.md`

## Manual environment probes
- command: `pwd`
  - result: `/home/zhangzhiwei/homework/shenduxuexi `
- command: `uname -a`
  - result: recorded in `logs/system_info.txt`
- command: `nvidia-smi`
  - result: sandboxed call failed with driver communication error; escalated call succeeded and showed 3 x NVIDIA RTX A6000, recorded in `logs/system_info.txt`
- command: `nvcc --version`
  - result: CUDA 12.4, recorded in `logs/system_info.txt`
- command: `/home/zhangzhiwei/miniconda3/bin/conda env list`
  - result: PG exists, recorded in `logs/system_info.txt`
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python -c ...`
  - result: PG Python 3.12.9, no torch/trimesh/open3d/rembg, recorded in `logs/system_info.txt`
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n base python -c ...`
  - result: PyTorch 2.6.0+cu124 can access CUDA when run with GPU permissions, recorded in `logs/system_info.txt`


## object_A_prepare_images
- start: 2026-06-22T09:23:58+00:00
- end: 2026-06-22T09:24:03+00:00
- git commit: no-git
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/extract_frames.py `
- output path: `outputs/task1/object_A_frame_stats.json`
- log: `logs/object_A_prepare_images_20260622_092358.log`
- success: yes
- duration_sec: 5
- gpu note: see nvidia-smi snapshots in the log

## object_C_preprocess_rgba
- start: 2026-06-22T09:24:09+00:00
- end: 2026-06-22T09:24:41+00:00
- git commit: no-git
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/preprocess_object_c.py `
- output path: `outputs/task1/object_C_image3d/cup_rgba.png`
- log: `logs/object_C_preprocess_rgba_20260622_092409.log`
- success: yes
- duration_sec: 32
- gpu note: see nvidia-smi snapshots in the log

## object_C_preprocess_rgba_v2
- start: 2026-06-22T09:25:44+00:00
- end: 2026-06-22T09:26:16+00:00
- git commit: no-git
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/preprocess_object_c.py `
- output path: `outputs/task1/object_C_image3d/cup_rgba.png`
- log: `logs/object_C_preprocess_rgba_v2_20260622_092544.log`
- success: yes
- duration_sec: 32
- gpu note: see nvidia-smi snapshots in the log

## object_C_preprocess_rgba_v3
- start: 2026-06-22T09:27:05+00:00
- end: 2026-06-22T09:27:56+00:00
- git commit: no-git
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/preprocess_object_c.py `
- output path: `outputs/task1/object_C_image3d/cup_rgba.png`
- log: `logs/object_C_preprocess_rgba_v3_20260622_092705.log`
- success: yes
- duration_sec: 51
- gpu note: see nvidia-smi snapshots in the log

## object_A_colmap_feature_extract
- start: 2026-06-22T09:28:18+00:00
- end: 2026-06-22T09:28:21+00:00
- git commit: no-git
- command: `colmap feature_extractor --database_path data/object_A/database.db --image_path data/object_A/input --ImageReader.single_camera 1 --SiftExtraction.use_gpu 0 `
- output path: `data/object_A/database.db`
- log: `logs/object_A_colmap_feature_extract_20260622_092818.log`
- success: yes
- duration_sec: 3
- gpu note: see nvidia-smi snapshots in the log

## object_A_colmap_exhaustive_match
- start: 2026-06-22T09:28:27+00:00
- end: 2026-06-22T09:28:29+00:00
- git commit: no-git
- command: `colmap exhaustive_matcher --database_path data/object_A/database.db --SiftMatching.use_gpu 0 `
- output path: `data/object_A/database.db`
- log: `logs/object_A_colmap_exhaustive_match_20260622_092827.log`
- success: yes
- duration_sec: 2
- gpu note: see nvidia-smi snapshots in the log

## object_A_colmap_mapper
- start: 2026-06-22T09:28:35+00:00
- end: 2026-06-22T09:29:47+00:00
- git commit: no-git
- command: `colmap mapper --database_path data/object_A/database.db --image_path data/object_A/input --output_path data/object_A/sparse `
- output path: `data/object_A/sparse`
- log: `logs/object_A_colmap_mapper_20260622_092835.log`
- success: yes
- duration_sec: 72
- gpu note: see nvidia-smi snapshots in the log

## object_A_prepare_images_all
- start: 2026-06-22T09:30:17+00:00
- end: 2026-06-22T09:30:20+00:00
- git commit: no-git
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/extract_frames.py --output data/object_A_all/input --stats outputs/task1/object_A_frame_stats_all.json --blur-quantile 0.0 --similarity-threshold 0.9999 `
- output path: `outputs/task1/object_A_frame_stats_all.json`
- log: `logs/object_A_prepare_images_all_20260622_093017.log`
- success: yes
- duration_sec: 3
- gpu note: see nvidia-smi snapshots in the log

## object_A_all_colmap_feature_extract
- start: 2026-06-22T09:30:30+00:00
- end: 2026-06-22T09:30:33+00:00
- git commit: no-git
- command: `colmap feature_extractor --database_path data/object_A_all/database.db --image_path data/object_A_all/input --ImageReader.single_camera 1 --SiftExtraction.use_gpu 0 `
- output path: `data/object_A_all/database.db`
- log: `logs/object_A_all_colmap_feature_extract_20260622_093030.log`
- success: yes
- duration_sec: 3
- gpu note: see nvidia-smi snapshots in the log

## object_A_all_colmap_exhaustive_match
- start: 2026-06-22T09:30:38+00:00
- end: 2026-06-22T09:30:40+00:00
- git commit: no-git
- command: `colmap exhaustive_matcher --database_path data/object_A_all/database.db --SiftMatching.use_gpu 0 `
- output path: `data/object_A_all/database.db`
- log: `logs/object_A_all_colmap_exhaustive_match_20260622_093038.log`
- success: yes
- duration_sec: 2
- gpu note: see nvidia-smi snapshots in the log

## object_A_all_colmap_mapper_relaxed
- start: 2026-06-22T09:30:47+00:00
- end: 2026-06-22T09:33:09+00:00
- git commit: no-git
- command: `colmap mapper --database_path data/object_A_all/database.db --image_path data/object_A_all/input --output_path data/object_A_all/sparse_relaxed --Mapper.init_min_num_inliers 15 --Mapper.abs_pose_min_num_inliers 15 --Mapper.init_min_tri_angle 1.0 --Mapper.filter_max_reproj_error 8.0 `
- output path: `data/object_A_all/sparse_relaxed`
- log: `logs/object_A_all_colmap_mapper_relaxed_20260622_093047.log`
- success: yes
- duration_sec: 142
- gpu note: see nvidia-smi snapshots in the log

## setup_third_party
- start: 2026-06-22T09:33:40+00:00
- end: 2026-06-22T09:34:50+00:00
- git commit: no-git
- command: `bash scripts/setup_third_party.sh `
- output path: `third_party`
- log: `logs/setup_third_party_20260622_093340.log`
- success: no
- duration_sec: 70
- gpu note: see nvidia-smi snapshots in the log

## setup_third_party_retry
- start: 2026-06-22T09:35:29+00:00
- end: 2026-06-22T09:38:39+00:00
- git commit: no-git
- command: `bash scripts/setup_third_party.sh `
- output path: `third_party`
- log: `logs/setup_third_party_retry_20260622_093529.log`
- success: yes
- duration_sec: 190
- gpu note: see nvidia-smi snapshots in the log

## calvin_hf_metadata_probe
- start: 2026-06-22T09:40:27+00:00
- end: 2026-06-22T09:40:38+00:00
- git commit: no-git
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/prepare_calvin_splits.py `
- output path: `outputs/task2/calvin_split_summary.json`
- log: `logs/calvin_hf_metadata_probe_20260622_094027.log`
- success: no
- duration_sec: 11
- gpu note: see nvidia-smi snapshots in the log

## install_3dgs_local_deps
- start: 2026-06-22T09:41:31+00:00
- end: 2026-06-22T09:44:52+00:00
- git commit: no-git
- command: `bash -lc mkdir\ -p\ local_pkgs/3dgs\ \&\&\ PIP_CACHE_DIR=/tmp/hw3_pip_cache\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ -m\ pip\ install\ --upgrade\ --target\ local_pkgs/3dgs\ plyfile\ ./third_party/gaussian-splatting/submodules/diff-gaussian-rasterization\ ./third_party/gaussian-splatting/submodules/simple-knn\ ./third_party/gaussian-splatting/submodules/fused-ssim `
- output path: `local_pkgs/3dgs`
- log: `logs/install_3dgs_local_deps_20260622_094131.log`
- success: yes
- duration_sec: 201
- gpu note: see nvidia-smi snapshots in the log

## object_A_official_3dgs_convert_all
- start: 2026-06-22T09:46:52+00:00
- end: 2026-06-22T09:47:01+00:00
- git commit: no-git
- command: `bash -lc cd\ third_party/gaussian-splatting\ \&\&\ PYTHONPATH=\"/home/zhangzhiwei/homework/shenduxuexi\ /hw3_spatial_intelligence_project/local_pkgs/3dgs\"\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ convert.py\ -s\ ../../data/object_A_all\ --resize\ --no_gpu `
- output path: `data/object_A_all/sparse`
- log: `logs/object_A_official_3dgs_convert_all_20260622_094652.log`
- success: yes
- duration_sec: 9
- gpu note: see nvidia-smi snapshots in the log

## object_A_official_3dgs_convert_patched
- start: 2026-06-22T09:48:01+00:00
- end: 2026-06-22T09:49:35+00:00
- git commit: no-git
- command: `bash -lc cd\ third_party/gaussian-splatting\ \&\&\ PYTHONPATH=\"/home/zhangzhiwei/homework/shenduxuexi\ /hw3_spatial_intelligence_project/local_pkgs/3dgs\"\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ convert.py\ -s\ ../../data/object_A_all_3dgs\ --resize\ --no_gpu `
- output path: `data/object_A_all_3dgs/sparse`
- log: `logs/object_A_official_3dgs_convert_patched_20260622_094801.log`
- success: no
- duration_sec: 94
- gpu note: see nvidia-smi snapshots in the log

## install_lerobot_light_deps
- start: 2026-06-22T09:50:09+00:00
- end: 2026-06-22T09:51:02+00:00
- git commit: no-git
- command: `bash -lc mkdir\ -p\ local_pkgs/lerobot\ \&\&\ PIP_CACHE_DIR=/tmp/hw3_pip_cache\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ -m\ pip\ install\ --upgrade\ --target\ local_pkgs/lerobot\ termcolor\ draccus\ safetensors\ gymnasium\ einops\ jsonlines\ opencv-python-headless\ pandas\ pyarrow\ requests\ packaging\ tqdm `
- output path: `local_pkgs/lerobot`
- log: `logs/install_lerobot_light_deps_20260622_095009.log`
- success: yes
- duration_sec: 53
- gpu note: see nvidia-smi snapshots in the log

## install_lerobot_hf_deps
- start: 2026-06-22T09:51:33+00:00
- end: 2026-06-22T09:52:18+00:00
- git commit: no-git
- command: `bash -lc PIP_CACHE_DIR=/tmp/hw3_pip_cache\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ -m\ pip\ install\ --upgrade\ --target\ local_pkgs/lerobot\ huggingface-hub\ datasets `
- output path: `local_pkgs/lerobot`
- log: `logs/install_lerobot_hf_deps_20260622_095133.log`
- success: yes
- duration_sec: 45
- gpu note: see nvidia-smi snapshots in the log

## install_lerobot_av_dep
- start: 2026-06-22T09:52:46+00:00
- end: 2026-06-22T09:52:59+00:00
- git commit: no-git
- command: `bash -lc PIP_CACHE_DIR=/tmp/hw3_pip_cache\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ -m\ pip\ install\ --upgrade\ --target\ local_pkgs/lerobot\ av `
- output path: `local_pkgs/lerobot`
- log: `logs/install_lerobot_av_dep_20260622_095246.log`
- success: yes
- duration_sec: 13
- gpu note: see nvidia-smi snapshots in the log

## install_lerobot_av15_dep
- start: 2026-06-22T09:53:29+00:00
- end: 2026-06-22T09:53:41+00:00
- git commit: no-git
- command: `bash -lc PIP_CACHE_DIR=/tmp/hw3_pip_cache\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ -m\ pip\ install\ --upgrade\ --force-reinstall\ --target\ local_pkgs/lerobot\ \"av\>=15.0.0\,\<16.0.0\" `
- output path: `local_pkgs/lerobot`
- log: `logs/install_lerobot_av15_dep_20260622_095329.log`
- success: yes
- duration_sec: 12
- gpu note: see nvidia-smi snapshots in the log

## lerobot_train_help
- start: 2026-06-22T09:54:43+00:00
- end: 2026-06-22T09:54:52+00:00
- git commit: no-git
- command: `bash -lc PYTHONPATH=\"/home/zhangzhiwei/homework/shenduxuexi\ /hw3_spatial_intelligence_project/local_pkgs/lerobot:/home/zhangzhiwei/homework/shenduxuexi\ /hw3_spatial_intelligence_project/third_party/lerobot/src\"\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ -m\ lerobot.scripts.lerobot_train\ --help\ \>\ logs/lerobot_train_help.txt `
- output path: `logs/lerobot_train_help.txt`
- log: `logs/lerobot_train_help_20260622_095443.log`
- success: yes
- duration_sec: 9
- gpu note: see nvidia-smi snapshots in the log

## object_A_video_extract_frames
- start: 2026-06-22T10:08:03+00:00
- end: 2026-06-22T10:08:09+00:00
- git commit: ac5cb56
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/extract_frames.py --input inputs/object_A/A_video.mp4 --output data/object_A_video/input --stats outputs/task1/object_A_video_frame_stats.json --fps 6 --max-width 1000 --blur-quantile 0.03 --similarity-threshold 0.999 `
- output path: `outputs/task1/object_A_video_frame_stats.json`
- log: `logs/object_A_video_extract_frames_20260622_100803.log`
- success: yes
- duration_sec: 6
- gpu note: see nvidia-smi snapshots in the log

## object_A_video_3dgs_convert
- start: 2026-06-22T10:08:39+00:00
- end: 2026-06-22T10:13:08+00:00
- git commit: ac5cb56
- command: `bash -lc cd\ third_party/gaussian-splatting\ \&\&\ PYTHONPATH=\"/home/zhangzhiwei/homework/shenduxuexi\ /hw3_spatial_intelligence_project/local_pkgs/3dgs\"\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ convert.py\ -s\ ../../data/object_A_video\ --resize\ --no_gpu\ --magick_executable\ convert `
- output path: `data/object_A_video/sparse`
- log: `logs/object_A_video_3dgs_convert_20260622_100839.log`
- success: yes
- duration_sec: 269
- gpu note: see nvidia-smi snapshots in the log

## object_A_video_colmap_analyzer
- start: 2026-06-22T10:13:38+00:00
- end: 2026-06-22T10:13:39+00:00
- git commit: ac5cb56
- command: `bash -lc colmap\ model_analyzer\ --path\ data/object_A_video/sparse/0\ \|\ tee\ outputs/task1/object_A_video_colmap_stats.txt `
- output path: `outputs/task1/object_A_video_colmap_stats.txt`
- log: `logs/object_A_video_colmap_analyzer_20260622_101338.log`
- success: yes
- duration_sec: 1
- gpu note: see nvidia-smi snapshots in the log

## object_A_video_3dgs_train_1000
- start: 2026-06-22T10:13:58+00:00
- end: 2026-06-22T10:14:17+00:00
- git commit: ac5cb56
- command: `bash -lc cd\ third_party/gaussian-splatting\ \&\&\ CUDA_VISIBLE_DEVICES=1\ PYTHONPATH=\"/home/zhangzhiwei/homework/shenduxuexi\ /hw3_spatial_intelligence_project/local_pkgs/3dgs\"\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ train.py\ -s\ ../../data/object_A_video\ -m\ ../../outputs/task1/object_A_3dgs\ --eval\ -r\ 2\ --iterations\ 1000\ --save_iterations\ 1000\ --test_iterations\ 1000\ --disable_viewer `
- output path: `outputs/task1/object_A_3dgs`
- log: `logs/object_A_video_3dgs_train_1000_20260622_101358.log`
- success: yes
- duration_sec: 19
- gpu note: see nvidia-smi snapshots in the log

## object_A_video_3dgs_render
- start: 2026-06-22T10:14:43+00:00
- end: 2026-06-22T10:14:58+00:00
- git commit: ac5cb56
- command: `bash -lc cd\ third_party/gaussian-splatting\ \&\&\ CUDA_VISIBLE_DEVICES=1\ PYTHONPATH=\"/home/zhangzhiwei/homework/shenduxuexi\ /hw3_spatial_intelligence_project/local_pkgs/3dgs\"\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ render.py\ -m\ ../../outputs/task1/object_A_3dgs\ --iteration\ 1000 `
- output path: `outputs/task1/object_A_3dgs/test`
- log: `logs/object_A_video_3dgs_render_20260622_101443.log`
- success: yes
- duration_sec: 15
- gpu note: see nvidia-smi snapshots in the log

## object_A_video_3dgs_metrics
- start: 2026-06-22T10:15:12+00:00
- end: 2026-06-22T10:16:30+00:00
- git commit: ac5cb56
- command: `bash -lc cd\ third_party/gaussian-splatting\ \&\&\ CUDA_VISIBLE_DEVICES=1\ PYTHONPATH=\"/home/zhangzhiwei/homework/shenduxuexi\ /hw3_spatial_intelligence_project/local_pkgs/3dgs\"\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ metrics.py\ -m\ ../../outputs/task1/object_A_3dgs `
- output path: `outputs/task1/object_A_3dgs/results.json`
- log: `logs/object_A_video_3dgs_metrics_20260622_101512.log`
- success: yes
- duration_sec: 78
- gpu note: see nvidia-smi snapshots in the log

## calvin_hf_metadata_probe_known_splits
- start: 2026-06-22T10:19:46+00:00
- end: 2026-06-22T10:19:56+00:00
- git commit: ac5cb56
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/prepare_calvin_splits.py --fallback-known-splits `
- output path: `outputs/task2/calvin_split_summary.json`
- log: `logs/calvin_hf_metadata_probe_known_splits_20260622_101946.log`
- success: yes
- duration_sec: 10
- gpu note: see nvidia-smi snapshots in the log

## proxy_assets_generate
- start: 2026-06-22T11:08:12+00:00
- end: 2026-06-22T11:08:15+00:00
- git commit: f813b12
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/generate_proxy_assets.py `
- output path: `outputs/task1/proxy_assets_meta.json`
- log: `logs/proxy_assets_generate_20260622_110812.log`
- success: yes
- duration_sec: 3
- gpu note: see nvidia-smi snapshots in the log

## background_counter_proxy_mesh_to_gaussian
- start: 2026-06-22T11:08:23+00:00
- end: 2026-06-22T11:08:26+00:00
- git commit: f813b12
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/mesh_to_gaussians.py --mesh outputs/task1/background_3dgs/counter_proxy/export/mesh.obj --output outputs/task1/background_3dgs/counter_proxy/point_cloud/iteration_0001/point_cloud.ply --sample-points 32000 --opacity 0.65 --seed 11 `
- output path: `outputs/task1/background_3dgs/counter_proxy/point_cloud/iteration_0001/point_cloud.ply`
- log: `logs/background_counter_proxy_mesh_to_gaussian_20260622_110823.log`
- success: yes
- duration_sec: 3
- gpu note: see nvidia-smi snapshots in the log

## merge_proxy_scene
- start: 2026-06-22T11:08:32+00:00
- end: 2026-06-22T11:08:45+00:00
- git commit: f813b12
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/merge_gaussians.py --config configs/placements.yaml --output outputs/task1/merged_scene/point_cloud.ply --workdir outputs/task1/merged_scene/intermediate `
- output path: `outputs/task1/merged_scene/point_cloud.ply`
- log: `logs/merge_proxy_scene_20260622_110832.log`
- success: yes
- duration_sec: 13
- gpu note: see nvidia-smi snapshots in the log

## merged_proxy_scene_render
- start: 2026-06-22T11:08:57+00:00
- end: 2026-06-22T11:09:13+00:00
- git commit: f813b12
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/render_merged_path.py --input outputs/task1/merged_scene/point_cloud.ply --video-output outputs/task1/videos/merged_scene_walkthrough.mp4 --frames-dir outputs/task1/merged_scene/preview_frames --num-frames 72 --width 960 --height 540 `
- output path: `outputs/task1/videos/merged_scene_walkthrough.mp4`
- log: `logs/merged_proxy_scene_render_20260622_110857.log`
- success: yes
- duration_sec: 16
- gpu note: see nvidia-smi snapshots in the log

## merged_proxy_scene_render_robust
- start: 2026-06-22T11:09:55+00:00
- end: 2026-06-22T11:10:12+00:00
- git commit: f813b12
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n PG python scripts/render_merged_path.py --input outputs/task1/merged_scene/point_cloud.ply --video-output outputs/task1/videos/merged_scene_walkthrough.mp4 --frames-dir outputs/task1/merged_scene/preview_frames --num-frames 72 --width 960 --height 540 --point-radius 2 `
- output path: `outputs/task1/videos/merged_scene_walkthrough.mp4`
- log: `logs/merged_proxy_scene_render_robust_20260622_110955.log`
- success: yes
- duration_sec: 17
- gpu note: see nvidia-smi snapshots in the log

## act_proxy_train
- start: 2026-06-22T11:12:16+00:00
- end: 2026-06-22T11:12:40+00:00
- git commit: f813b12
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n base python scripts/train_act_proxy.py --steps 800 --log-every 20 `
- output path: `outputs/task2/proxy_act_summary.json`
- log: `logs/act_proxy_train_20260622_111216.log`
- success: yes
- duration_sec: 24
- gpu note: see nvidia-smi snapshots in the log

## act_proxy_train_style_marker
- start: 2026-06-22T11:13:29+00:00
- end: 2026-06-22T11:13:43+00:00
- git commit: f813b12
- command: `/home/zhangzhiwei/miniconda3/bin/conda run -n base python scripts/train_act_proxy.py --steps 800 --log-every 20 `
- output path: `outputs/task2/proxy_act_summary.json`
- log: `logs/act_proxy_train_style_marker_20260622_111329.log`
- success: yes
- duration_sec: 14
- gpu note: see nvidia-smi snapshots in the log

## act_real_calvin_subset
- start: 2026-06-23T11:13:45+00:00
- end: 2026-06-23T11:13:55+00:00
- git commit: ea38ad6
- command: `env MPLCONFIGDIR=/tmp/hw3_mpl XDG_CACHE_HOME=/tmp/hw3_cache /home/zhangzhiwei/miniconda3/bin/conda run -n base python scripts/train_act_calvin_subset.py --episodes-per-split 4 --steps 40 --log-every 10 --chunk-size 8 --stride 4 --batch-size 16 `
- output path: `outputs/task2/real_calvin_subset_summary.json`
- log: `logs/act_real_calvin_subset_20260623_111345.log`
- success: yes
- duration_sec: 10
- gpu note: see nvidia-smi snapshots in the log

## background_counter_3dgs_train
- start: 2026-06-23T12:47:12+00:00
- end: 2026-06-23T12:48:07+00:00
- git commit: 4e0884f
- command: `bash -lc cd\ third_party/gaussian-splatting\ \&\&\ PYTHONPATH=\'/home/zhangzhiwei/homework/shenduxuexi\ /hw3_spatial_intelligence_project/local_pkgs/3dgs\'\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ train.py\ -s\ ../../data/mipnerf360/counter\ -m\ ../../outputs/task1/background_3dgs/counter\ --eval\ -r\ 8\ --iterations\ 1000\ --disable_viewer `
- output path: `outputs/task1/background_3dgs/counter`
- log: `logs/background_counter_3dgs_train_20260623_124712.log`
- success: yes
- duration_sec: 55
- gpu note: see nvidia-smi snapshots in the log

## background_counter_3dgs_render
- start: 2026-06-23T12:48:12+00:00
- end: 2026-06-23T12:48:59+00:00
- git commit: 4e0884f
- command: `bash -lc cd\ third_party/gaussian-splatting\ \&\&\ PYTHONPATH=\'/home/zhangzhiwei/homework/shenduxuexi\ /hw3_spatial_intelligence_project/local_pkgs/3dgs\'\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ render.py\ -m\ ../../outputs/task1/background_3dgs/counter `
- output path: `outputs/task1/background_3dgs/counter`
- log: `logs/background_counter_3dgs_render_20260623_124812.log`
- success: yes
- duration_sec: 47
- gpu note: see nvidia-smi snapshots in the log

## background_counter_3dgs_metrics
- start: 2026-06-23T12:48:59+00:00
- end: 2026-06-23T12:49:36+00:00
- git commit: 4e0884f
- command: `bash -lc cd\ third_party/gaussian-splatting\ \&\&\ PYTHONPATH=\'/home/zhangzhiwei/homework/shenduxuexi\ /hw3_spatial_intelligence_project/local_pkgs/3dgs\'\ /home/zhangzhiwei/miniconda3/bin/conda\ run\ -n\ base\ python\ metrics.py\ -m\ ../../outputs/task1/background_3dgs/counter `
- output path: `outputs/task1/background_3dgs/counter/results.json`
- log: `logs/background_counter_3dgs_metrics_20260623_124859.log`
- success: yes
- duration_sec: 37
- gpu note: see nvidia-smi snapshots in the log
