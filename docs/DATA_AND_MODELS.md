# 数据、模型与许可边界

更新：2026-10-30。返回[快速开始](QUICKSTART.md)。

## WILDTRACK

从[EPFL官方入口](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)取得数据并阅读包内README及适用条件。本仓库不提供数据、真实标注GT、标定或图像/视频副本。使用数据时应遵循包内条件并引用作者论文。

```text
data/Wildtrack_dataset/
  Image_subsets/C1/00000800.png ... 00001395.png
  Image_subsets/C2/00000800.png ... 00001395.png
  Image_subsets/C3/00000800.png ... 00001395.png
  calibrations/intrinsic_zero/intr_CVLab1.xml (以及2、3)
  calibrations/extrinsic/extr_CVLab1.xml (以及2、3)
  annotations_positions/00000800.json ...    # 仅评价需要
weights/yolo11n.pt
```

采用数据提供的1920×1080去畸变图像及intrinsic_zero。外参是世界到相机，t由厘米转米；positionID为480列、2.5cm网格，转换为`X=-3+0.025*(ID%480)`、`Y=-9+0.025*(ID//480)`。C1–C3是本项目使用的三台相机编号；地面评价区域ROI及物理时间步见[协议](EVALUATION.md)。官方原始视频帧率与这里选用每5个编号一次的0.5秒时刻不是同一概念。

引用：[Chavdarova等，WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection，CVPR 2018](https://arxiv.org/abs/1707.09299)。

## YOLO11n

使用[Ultralytics官方YOLO11模型说明](https://docs.ultralytics.com/models/yolo11/)中的预训练检测权重`yolo11n.pt`，由读者自行获取并放置。未训练或微调，不提供权重副本。原运行文件5,613,764字节；版本是Ultralytics8.3.253，person类0、imgsz640、conf0.25、iou0.7、FP32、cuda:0；相同文件名不自动保证权重内容相同。

本地实验实际权重的SHA256：`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。这是本地实验文件的校验值，并非官方公布的哈希。

软件包与模型适用条件参见[Ultralytics许可](https://www.ultralytics.com/license)及[固定版LICENSE](https://github.com/ultralytics/ultralytics/blob/v8.3.253/LICENSE)。本项目未取得Enterprise许可，也没有用项目AGPL声明替代模型/数据各自条件。

## 发布数值

`results/b3_reference`为三相机等权融合方案B3对应的预测缓存：检测框、置信度、脚点、投影、轨迹等数值，不含原图或GT文件。汇总表来自各轮实验，来源路径和可复核范围见[PROVENANCE](PROVENANCE.md)。数值可供阅读/回放核对；完整身份重评还需读者自备原GT和完整相应轨迹。
