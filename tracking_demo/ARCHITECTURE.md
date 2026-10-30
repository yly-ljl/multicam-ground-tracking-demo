# tracking_demo模块

更新：2026-10-30。返回[目录导航](../ARCHITECTURE.md)，按[代码路线](../docs/CODE_READING.md)阅读。全部Python文件来自当前源码，注释和排版原样保留。

| 模块 | 职责 |
|---|---|
| [geometry.py](geometry.py) | GroundCamera标定/单位/投影/ROI与grid_to_world。 |
| [fusion.py](fusion.py) | 确定相机顺序的完整成员门融合，组内等权。 |
| [tracking.py](tracking.py) | 物理时间FilterPy与Norfair原贪心包装。 |
| [pipeline.py](pipeline.py) | 原E001单相机完整预测与导出。 |
| [multicam.py](multicam.py) | 多相机缓存、融合、导出及原研究媒体。 |
| [improvement.py](improvement.py) | E003固定组敏感度加权候选及关闭候选重放。 |
| [association.py](association.py) | E005/E006匹配函数和局部Tracker子类，默认不启用。 |
| [temporal_comparison.py](temporal_comparison.py) | 从固定融合CSV独立重放时间候选；需相应研究缓存。 |
| [analysis.py](analysis.py) | GT独立几何诊断及原空间配对。 |
| [identity_evaluation.py](identity_evaluation.py) | 完整距离关系、整段身份与独立空间汇总。 |
| [association_diagnostics.py](association_diagnostics.py) | 原贪心实际阶段/候选/预测/选择的CPU记录器。 |
| [temporal_analysis.py](temporal_analysis.py) | E005指标、变化与门槛。 |
| [soft_assignment_analysis.py](soft_assignment_analysis.py) | E006独立候选评价及明确历史来源比较。 |
| [observation_diagnostics.py](observation_diagnostics.py) | E007精确父组、四集合、抽样、有限图证与人工记录核查。 |
| [render.py](render.py) | 原研究相机/BEV/按秒轨迹绘制。 |
| [showcase.py](showcase.py) | B3全部状态总览和同米制比例zoom。 |
| [demo_delivery.py](demo_delivery.py) | 对外play/replay/images及新目录保护。 |
| [__init__.py](__init__.py) | 包标记。 |

预测不读GT；评价/候选/诊断模块分离。后者的完整输入不全部随发布提供，详见配置和脚本前提。
