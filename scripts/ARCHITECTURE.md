# 脚本前提与副作用

更新：2026-10-30。返回[目录导航](../ARCHITECTURE.md)。run_cpu_tests是开发测试入口，build_public_figures从随附数值生成自绘图。研究脚本依赖各阶段完整档案，部分含硬编码路径或顶层执行，使用前按下表准备输入并检查输出。

| 脚本 | 前提与写出范围 |
|---|---|
| [build_phase5_media.py](build_phase5_media.py) | 需原预测视频及结果制作讲解/单页；仓库未附带原图媒体，不能在clone中直接运行。 |
| [build_public_figures.py](build_public_figures.py) | 自绘图脚本；读随附summary CSV和代码关系，只写docs/assets SVG；可指定发布树外的PNG预览目录，不读数据图像或模型。 |
| [check_dataset.py](check_dataset.py) | 原完整资产检查；需原数据和归档/清单，会写setup_check，不作为clone后的首次测试。 |
| [check_e006_regression.py](check_e006_regression.py) | 硬编码原E006配置及旧开发B3全轨迹/组；执行100tick CPU回放并写E006，依赖未发布完整缓存。 |
| [check_environment.py](check_environment.py) | 原环境/首帧检查；可涉及CUDA/真实YOLO推理并写setup_check，不是无数据CPU测试入口。 |
| [check_phase5_delivery.py](check_phase5_delivery.py) | 原历史交付审计；需当时保护清单、视频、性能等完整档案，并要求当时文件集合；不适用于整理后的发布树。 |
| [check_real_inputs.py](check_real_inputs.py) | 原环境/首帧检查；可涉及CUDA/真实YOLO推理并写setup_check，不是无数据CPU测试入口。 |
| [review_e006_cases.py](review_e006_cases.py) | 原阶段案例诊断脚本；依赖完整阶段缓存、GT/图像与历史来源，可能顶层直接运行并写旧阶段目录；不要为阅读而导入。 |
| [review_identity_cases.py](review_identity_cases.py) | 原阶段案例诊断脚本；依赖完整阶段缓存、GT/图像与历史来源，可能顶层直接运行并写旧阶段目录；不要为阅读而导入。 |
| [review_temporal_cases.py](review_temporal_cases.py) | 原阶段案例诊断脚本；依赖完整阶段缓存、GT/图像与历史来源，可能顶层直接运行并写旧阶段目录；不要为阅读而导入。 |
| [run_cpu_tests.py](run_cpu_tests.py) | 开发测试封装；不需数据运行46项；--with-calibration需3路标定。只创建运行outputs根。 |
| [run_e007_diagnostics.py](run_e007_diagnostics.py) | E007需原三段B2/B3融合/成员/轨迹、GT、全部距离/映射、旧关联日志和登记；render还需原图。写E007新诊断，不是本仓库四CSV的运行示例。 |

正式入门生成入口为根run_demo.py配public_demo_reference。仓库未附带数据、模型或完整历史档案。
