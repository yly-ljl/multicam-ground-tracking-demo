# 测试与外部资产

更新：2026-10-30。返回[目录导航](../ARCHITECTURE.md)。八份原测试逐字保留，共48项，不调用GPU检测。

| 文件 | 覆盖 |
|---|---|
| [test_core.py](test_core.py) | 门内匹配、物理时间/空生命周期；其中metric_projection_and_opencv需CVLab1标定。 |
| [test_fusion.py](test_fusion.py) | 成员唯一/同相机互斥、完整成员门、均值、历史时间。 |
| [test_improvement.py](test_improvement.py) | 加权/关闭候选；其中analytic_jacobian_matches_finite_difference需CVLab1/2/3标定。 |
| [test_identity.py](test_identity.py) | 手算身份、空帧、1m含边界、完整距离关系、缺标注错误；不读真实GT。 |
| [test_association.py](test_association.py) | E005严格门、最大配对目标、默认类与生命周期。 |
| [test_soft_assignment.py](test_soft_assignment.py) | E006单侧未匹配、同成本/矩形/禁边、新建与显式空间来源。 |
| [test_observation_diagnostics.py](test_observation_diagnostics.py) | E007父组/四集合/映射/固定采样及条件成员归属。 |
| [test_demo_entry.py](test_demo_entry.py) | 新输出必填、旧结果保护；要求outputs根存在。 |

推荐[run_cpu_tests.py](../scripts/run_cpu_tests.py)：默认先创建outputs根，明确排除两项外部标定测试，运行46项；--with-calibration包括全部48项。原测试文件未因资产缺省被删改。发布整理只验证默认46项，另两项标定测试本轮未运行。
