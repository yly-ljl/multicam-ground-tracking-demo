# 第三方、模型与数据声明

更新：2026-10-30。本项目新增源码/文档由yly-ljl按[GNU AGPL版本3](LICENSE)发布。根LICENSE为完整AGPL-3.0文本，来自本机固定Ultralytics发行包内同一规范文本；没有将第三方代码的版权改为项目作者。

本仓库通过依赖调用以下组件，没有附带wheel、CUDA运行库、第三方源码、模型权重或数据集。许可核对基于现有安装分发信息及其附带LICENSE，不只凭包名推断；完整48包的版本/元数据摘要见[DEPENDENCIES.csv](docs/DEPENDENCIES.csv)，核心分发许可原文保留在[third_party目录](docs/third_party/README.md)。表中主许可不能覆盖其wheel捆绑的全部第三方条款。

| 组件/固定版本 | 核对的许可与上游 |
|---|---|
| Ultralytics8.3.253、ultralytics-thop2.0.18 | 本地AGPL-3.0；[固定版LICENSE](https://github.com/ultralytics/ultralytics/blob/v8.3.253/LICENSE)，[官方许可与Enterprise方案](https://www.ultralytics.com/license)。本项目未购买商业许可。 |
| Norfair2.3.0 | BSD-3-Clause，[上游许可](https://github.com/tryolabs/norfair/blob/v2.3.0/LICENSE)。 |
| FilterPy1.4.5 | MIT，[上游许可](https://github.com/rlabbe/filterpy/blob/master/LICENSE)。 |
| NumPy1.26.4 / SciPy1.13.1 | BSD-3-Clause主体，本地LICENSE含捆绑组件通知；[NumPy](https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt)、[SciPy](https://github.com/scipy/scipy/blob/v1.13.1/LICENSE.txt)。 |
| opencv-python4.11.0.86 | Python打包层本地LICENSE为MIT；metadata标Apache2，对应OpenCV主体。wheel还有第三方通知，不简化为单一许可；[打包仓库](https://github.com/opencv/opencv-python)、[OpenCV4.11许可](https://github.com/opencv/opencv/blob/4.11.0/LICENSE)。 |
| PyTorch2.4.1+cu124 / torchvision0.19.1+cu124 | BSD主体及本地捆绑通知；[torch](https://github.com/pytorch/pytorch/blob/v2.4.1/LICENSE)、[vision](https://github.com/pytorch/vision/blob/v0.19.1/LICENSE)。CUDA发行可能含独立组件条件，项目源码许可不替代它们。 |
| motmetrics1.4.0 / pandas2.2.3 | MIT / BSD-3-Clause主体；[motmetrics](https://github.com/cheind/py-motmetrics)、[pandas](https://github.com/pandas-dev/pandas/blob/v2.2.3/LICENSE)。 |

**模型**：YOLO11n由读者从Ultralytics获取并遵守其适用条件；不因仓库私有或源码声明AGPL而改变第三方条件。**数据**：WILDTRACK由作者提供，包内要求论文引用；首传不分发原图、GT、标定、视频或裁图。参考数值是项目预测/汇总，不声称获得了数据图像再分发授权。详见[数据与模型](docs/DATA_AND_MODELS.md)。

官方来源查阅日期2026-10-30；实际安装版本与包内许可文本才是本次复核的固定对象。后续升级依赖应重新核对。首传选择AGPL并不表示所有外部资产都改用AGPL。
