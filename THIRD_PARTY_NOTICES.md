# 第三方、模型与数据声明

更新：2026-10-30。本项目新增源码/文档由yly-ljl按[GNU AGPL版本3](LICENSE)发布。第三方组件保留各自的版权与许可。

本仓库通过依赖调用以下组件，没有附带wheel、CUDA运行库、第三方源码、模型权重或数据集。许可信息来自对应发行包的元数据及附带LICENSE；完整48包的版本/元数据摘要见[DEPENDENCIES.csv](docs/DEPENDENCIES.csv)，核心分发许可原文保留在[third_party目录](docs/third_party/README.md)。表中主许可不能覆盖其wheel捆绑的全部第三方条款。

| 组件/固定版本 | 核对的许可与上游 |
|---|---|
| Ultralytics8.3.253、ultralytics-thop2.0.18 | 本地AGPL-3.0；[固定版LICENSE](https://github.com/ultralytics/ultralytics/blob/v8.3.253/LICENSE)，[官方许可与Enterprise方案](https://www.ultralytics.com/license)。 |
| Norfair2.3.0 | BSD-3-Clause，[上游许可](https://github.com/tryolabs/norfair/blob/v2.3.0/LICENSE)。 |
| FilterPy1.4.5 | MIT，[上游许可](https://github.com/rlabbe/filterpy/blob/master/LICENSE)。 |
| NumPy1.26.4 / SciPy1.13.1 | BSD-3-Clause主体，本地LICENSE含捆绑组件通知；[NumPy](https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt)、[SciPy](https://github.com/scipy/scipy/blob/v1.13.1/LICENSE.txt)。 |
| opencv-python4.11.0.86 | Python打包层本地LICENSE为MIT；metadata标Apache2，对应OpenCV主体。wheel还有第三方通知，不简化为单一许可；[打包仓库](https://github.com/opencv/opencv-python)、[OpenCV4.11许可](https://github.com/opencv/opencv/blob/4.11.0/LICENSE)。 |
| PyTorch2.4.1+cu124 / torchvision0.19.1+cu124 | BSD主体及本地捆绑通知；[torch](https://github.com/pytorch/pytorch/blob/v2.4.1/LICENSE)、[vision](https://github.com/pytorch/vision/blob/v0.19.1/LICENSE)。CUDA发行可能含独立组件条件，项目源码许可不替代它们。 |
| motmetrics1.4.0 / pandas2.2.3 | MIT / BSD-3-Clause主体；[motmetrics](https://github.com/cheind/py-motmetrics)、[pandas](https://github.com/pandas-dev/pandas/blob/v2.2.3/LICENSE)。 |

**模型**：YOLO11n由读者从Ultralytics获取并遵守其适用条件。**数据**：WILDTRACK由作者提供，包内要求论文引用；仓库不分发原图、GT、标定、视频或裁图。项目源码的AGPL许可不替代模型和数据的适用条件。详见[数据与模型](docs/DATA_AND_MODELS.md)。

来源查阅日期：2026-10-30。上表针对固定依赖版本；升级依赖时应重新检查相应许可。
