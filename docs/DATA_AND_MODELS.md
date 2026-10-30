# 数据、模型与许可边界

更新：2026-10-30。返回[快速开始](QUICKSTART.md)。

## WILDTRACK

从[EPFL官方入口](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)取得数据并阅读包内README及适用条件。仓库没有再托管数据、GT、标定或媒体；官方页面提供下载，不将“可下载”理解为项目可以重新分发图像/视频。包内README要求引用作者论文。首传没有原图截图、60秒/80秒视频或48窗裁图。

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

采用数据提供的1920×1080去畸变图像及intrinsic_zero。外参是世界到相机，t由厘米转米；positionID为480列、2.5cm网格，转换为`X=-3+0.025*(ID%480)`、`Y=-9+0.025*(ID//480)`。本项目仅用C1–C3，固定半开ROI和物理时间步见[协议](EVALUATION.md)。官方原始视频帧率与这里选用每5个编号一次的0.5秒时刻不是同一概念。

引用：[Chavdarova等，WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection，CVPR 2018](https://arxiv.org/abs/1707.09299)。项目引用数据不表示数据作者认可本demo成绩。

## YOLO11n

使用[Ultralytics官方YOLO11模型说明](https://docs.ultralytics.com/models/yolo11/)中的预训练检测权重`yolo11n.pt`，由读者自行获取并放置。未训练或微调，不提供权重副本。原运行文件5,613,764字节；版本是Ultralytics8.3.253，person类0、imgsz640、conf0.25、iou0.7、FP32、cuda:0；相同文件名不自动保证权重内容相同。

本地实验实际权重的SHA256：`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。这是维护者对原5,613,764字节文件的只读校验值，不称为官方公布哈希；本次整理没有下载、替换或重新加载权重。

软件包与模型适用条件参见[Ultralytics许可](https://www.ultralytics.com/license)及[固定版LICENSE](https://github.com/ultralytics/ultralytics/blob/v8.3.253/LICENSE)。本项目未取得Enterprise许可，也没有用项目AGPL声明替代模型/数据各自条件。

## 发布数值

`results/b3_reference`为已验收预测缓存：检测框、置信度、脚点、投影、轨迹等数值，不含原图或GT文件。汇总表来自本地已验收研究，原路径作为PROVENANCE中的来源标识，不表示这些内部目录随仓库提供。数值可供阅读/回放核对；完整身份重评还需读者自备原GT和完整相应轨迹。
