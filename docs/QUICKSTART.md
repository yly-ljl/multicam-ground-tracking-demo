# 快速开始

更新：2026-10-30。返回[首页](../README.md)。阅读结果无需安装环境；生成视频可选择从原图推理（images）或从预测缓存回放（replay）。两者均需自行准备原图和标定。

## 环境

实际验证环境为Windows x64、Python3.12.6；PyTorch2.4.1+cu124、torchvision0.19.1+cu124、Ultralytics8.3.253、NumPy1.26.4、SciPy1.13.1、OpenCV4.11.0.86、Norfair2.3.0、FilterPy1.4.5、motmetrics1.4.0。requirements列出实际固定发行版本，不是跨平台依赖文件。

```powershell
git clone https://github.com/yly-ljl/multicam-ground-tracking-demo.git
cd multicam-ground-tracking-demo
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install setuptools==80.9.0 wheel==0.45.1
.venv/Scripts/python.exe -m pip install --no-build-isolation -r requirements.txt
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe run_demo.py --help
```

私有仓库需账号访问权限。这些命令用于重建既有研究环境；全新机器安装尚未验证。requirements中的torch/torchvision URL绑定CPython3.12/Windows amd64/CUDA12.4发行；其它系统或Python版本需另做适配。当前images图像推理需要支持CUDA的NVIDIA GPU；replay读取已有预测缓存，可在CPU上运行，但仍需要原图和标定。

## 准备外部资产

按[数据与模型](DATA_AND_MODELS.md)自行取得去畸变图像及标定，放到`data/Wildtrack_dataset/`，YOLO11n权重放`weights/yolo11n.pt`。`public_demo_reference`使用C1/C2/C3三路相机的等权融合方案（B3）。帧编号800–1395、步长5，共120个时刻、360张图，对应已使用的对照片段：它原在E003用于验证，后已用于分析比较。`frame_stop=1400`不包含1400。

`images`和`replay`展示不读真实标注GT；只需相应三路图像和标定，images另需权重。重算评价另外需要annotations_positions。1400–1995仍未用于本项目实验。

## 从图像生成demo

```powershell
.venv/Scripts/python.exe run_demo.py images --config configs/public_demo_reference.json --output outputs/demo_images_01
Invoke-Item .\outputs\demo_images_01\pass_1\prediction.mp4
```

本入口读取原图、标定、权重，重新检测/投影/融合/跟踪/渲染，不读取旧outputs中的观测或轨迹。配置内reference_tracks仅为读者核对提供位置，不参与预测；images不使用observation_sources。写出在`pass_1/`，含views、融合组/成员、tracks、ticks、timings和run，外层performance.json保存计时。2fps是源采样/播放速度，不是推理吞吐。

## 用提供的预测缓存复现

```powershell
.venv/Scripts/python.exe run_demo.py replay --config configs/public_demo_reference.json --output outputs/demo_cache_01
.venv/Scripts/python.exe run_demo.py play --config configs/public_demo_reference.json --inspect
.venv/Scripts/python.exe run_demo.py play --config configs/public_demo_reference.json
```

replay读取`results/b3_reference/C1_observations.csv`等三份缓存，重新融合/默认贪心跟踪及绘图，不加载YOLO或进行GPU检测。它仍读取原图作背景并读取标定。输出prediction.mp4直接在新目录；默认play配置对应上述demo_cache_01。若换输出名，复制public配置并相应修改video_artifact再播放，或直接Invoke-Item。

每次使用**不存在的新outputs子目录**，已有目录会被拒绝。新配置可以另存，不修改原研究配置或参考CSV。参考缓存的内容与范围见[缓存说明](../results/b3_reference/README.md)。

## CPU测试

```powershell
.venv/Scripts/python.exe scripts/run_cpu_tests.py
# 已放置C1/C2/C3标定后，才可包括另两项标定测试
.venv/Scripts/python.exe scripts/run_cpu_tests.py --with-calibration
```

默认运行46项合成/逻辑测试；另外两项`GroundCamera`投影/Jacobian测试需外部标定，使用`--with-calibration`可包括它们。两种模式均不检测、训练或运行完整跟踪序列。原unittest discover仍保留，但无标定时会在这两项失败；其中输出保护测试要求outputs根目录存在，封装会创建它。

## 历史研究入口

原configs及scripts原样保留，但很多路径指向未随仓库提供的完整研究outputs/登记/逐帧距离导出。它们是研究与代码阅读入口，**不保证clone后直接运行**。四个参考CSV不能重建E001–E007所有中间证据。见[配置前提](../configs/ARCHITECTURE.md)、[脚本前提](../scripts/ARCHITECTURE.md)、[数值范围](../results/b3_reference/README.md)。旧evaluate/identity/trace可能写配置中的原输出，不应作为普通入门的“全部重跑”命令。
