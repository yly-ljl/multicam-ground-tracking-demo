# 快速开始

更新：2026-10-30。返回[首页](../README.md)。先选择“阅读”“images”“replay”；克隆不等于取得外部资产。

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

私有仓库需账号访问权限。上面是环境重建说明，本次发布整理没有执行安装，也未验证干净机器安装。requirements中的torch/torchvision URL绑定CPython3.12/Windows amd64/CUDA12.4发行；其它系统或Python版本需另做适配。当前`images`实现显式调用CUDA同步并检查CUDA模型，不支持仅改device就宣称CPU图像推理可用。

## 准备外部资产

按[数据与模型](DATA_AND_MODELS.md)自行取得去畸变图像及标定，放到`data/Wildtrack_dataset/`，YOLO11n权重放`weights/yolo11n.pt`。public_demo_reference为800–1395/步长5，共120个三路时刻、360张图。`frame_stop=1400`不包含1400。

`images`和`replay`展示不读GT；只需相应三路图像和C1/C2/C3标定。重算评价另外需要annotations_positions。不要将1400–1995用于本项目当前实验；该段仍保留。

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

每次使用**不存在的新outputs子目录**，已有目录会被拒绝。新配置可以另存，不修改原研究配置或参考CSV。本轮使用发布副本代码做了一次120tick回放：3569组、4956状态，全部轨迹字段与参考一致，最大数值差0；没有再次执行images或旧性能实验。

## CPU测试

```powershell
.venv/Scripts/python.exe scripts/run_cpu_tests.py
# 已放置C1/C2/C3标定后，才可包括另两项标定测试
.venv/Scripts/python.exe scripts/run_cpu_tests.py --with-calibration
```

封装不改原测试，仅明确选择：默认46项合成/逻辑测试本轮通过；两项`GroundCamera`投影/Jacobian测试需外部标定，本轮未运行。两种模式均不检测、训练或运行完整跟踪序列。原unittest discover仍保留，但无标定时会在这两项失败；其中输出保护测试要求outputs根目录存在，封装会创建它。

## 历史研究入口

原configs及scripts原样保留，但很多路径指向未随首传提供的完整研究outputs/登记/逐帧距离导出。它们是研究与代码阅读入口，**不保证clone后直接运行**。四个参考CSV不能重建E001–E007所有中间证据。见[配置前提](../configs/ARCHITECTURE.md)、[脚本前提](../scripts/ARCHITECTURE.md)、[数值范围](../results/b3_reference/README.md)。旧evaluate/identity/trace可能写配置中的原输出，不应作为普通入门的“全部重跑”命令。
