# 配置与前置条件

更新：2026-10-30。返回[目录导航](../ARCHITECTURE.md)。原11份JSON原样保留，public配置单列新增；参数不是按发布结果重新调优。

| 配置 | 前提与用途 |
|---|---|
| [c1_baseline.json](c1_baseline.json) | E001 C1；需完整图像/标定/权重，原模式会写outputs/E001_c1_baseline。 |
| [demo_benchmark.json](demo_benchmark.json) | 原展示/性能配置；原样保留，缓存/视频/reference_tracks指向未发布研究outputs，不能替代public配置。 |
| [demo_reference.json](demo_reference.json) | 原展示/性能配置；原样保留，缓存/视频/reference_tracks指向未发布研究outputs，不能替代public配置。 |
| [e003_development.json](e003_development.json) | 原实验配置原件；依赖该阶段完整观测/轨迹/GT/评价或登记档案，首传仅有摘要，不保证直接执行。 |
| [e003_validation.json](e003_validation.json) | 原实验配置原件；依赖该阶段完整观测/轨迹/GT/评价或登记档案，首传仅有摘要，不保证直接执行。 |
| [e004_identity.json](e004_identity.json) | 原实验配置原件；依赖该阶段完整观测/轨迹/GT/评价或登记档案，首传仅有摘要，不保证直接执行。 |
| [e004_new_development.json](e004_new_development.json) | 原实验配置原件；依赖该阶段完整观测/轨迹/GT/评价或登记档案，首传仅有摘要，不保证直接执行。 |
| [e005_temporal.json](e005_temporal.json) | 原实验配置原件；依赖该阶段完整观测/轨迹/GT/评价或登记档案，首传仅有摘要，不保证直接执行。 |
| [e006_soft_assignment.json](e006_soft_assignment.json) | 原实验配置原件；依赖该阶段完整观测/轨迹/GT/评价或登记档案，首传仅有摘要，不保证直接执行。 |
| [e007_observation_diagnostics.json](e007_observation_diagnostics.json) | 原实验配置原件；依赖该阶段完整观测/轨迹/GT/评价或登记档案，首传仅有摘要，不保证直接执行。 |
| [multicam_baseline.json](multicam_baseline.json) | 原实验配置原件；依赖该阶段完整观测/轨迹/GT/评价或登记档案，首传仅有摘要，不保证直接执行。 |
| [public_demo_reference.json](public_demo_reference.json) | 对外800–1395 B3。发布CSV＋读者原图/标定；images另需权重。支持快速开始。 |

frame_stop不包含上界；data_root以发布根目录解析。旧JSON中的outputs来源是原研究布局，不代表文件随首传提供。evaluate/identity/trace等会写配置输出；请勿直接执行all覆盖自己的历史结果。
