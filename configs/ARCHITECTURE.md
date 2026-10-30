# 配置与前置条件

更新：2026-10-30。返回[目录导航](../ARCHITECTURE.md)。入门使用public_demo_reference；其余配置保留各轮研究所用的输入/输出布局。C1是单相机；B2为C1+C2；B3为C1+C2+C3等权方案。

| 配置 | 用途 |
|---|---|
| [public_demo_reference.json](public_demo_reference.json) | 三相机演示，800–1395。replay使用仓库CSV和外部原图/标定；images另需权重。见[快速开始](../docs/QUICKSTART.md)。 |
| [c1_baseline.json](c1_baseline.json) | E001单相机C1，0–495；检测、投影、跟踪与原可见标注评价，输出到outputs/E001_c1_baseline。 |
| [multicam_baseline.json](multicam_baseline.json) | E002，0–495；比较单/双/三相机等权融合与地面位置结果。 |
| [e003_development.json](e003_development.json) | E003旧开发0–495；固定成员比较双/三相机等权与投影敏感度加权。 |
| [e003_validation.json](e003_validation.json) | E003当时的验证800–1395，现为已使用的对照片段；同样比较等权与加权。 |
| [e004_new_development.json](e004_new_development.json) | E004新开发500–795；生成三路观测及单/双/三相机条件，为身份比较补充片段。 |
| [e004_identity.json](e004_identity.json) | E004；读取三个片段各方案的轨迹与真实标注，统一计算身份与空间指标。 |
| [e005_temporal.json](e005_temporal.json) | E005；固定三段B3融合组，对比原贪心与优先最大配对数的GLOBAL时间关联。 |
| [e006_soft_assignment.json](e006_soft_assignment.json) | E006；固定同帧组运行SOFT有限未匹配代价候选，并与保存的B3/GLOBAL结果比较。 |
| [e007_observation_diagnostics.json](e007_observation_diagnostics.json) | E007；读取三段B2/B3组、成员、轨迹及评价导出，拆分新增/损失并选取诊断窗口。 |
| [demo_benchmark.json](demo_benchmark.json) | 原演示性能配置，0–95的小片段；缓存和结果路径指向完整研究目录。 |
| [demo_reference.json](demo_reference.json) | 原800–1395演示配置；与public配置用途相近，但缓存/视频路径仍指向完整研究目录。 |

旧/新开发仅表示同场景片段的使用顺序。800–1395原在E003用于验证，后用于分析比较，不是未使用测试段；1400–1995仍未用于实验。原配置字段与路径保持研究时的名称。

原研究入口按阶段需要完整观测、融合组、轨迹、真实标注和评价导出，诊断还可能需要关联日志与图像。仓库只附带部分缓存和汇总，不能仅凭这些文件直接重跑所有配置。具体写出行为见[脚本导航](../scripts/ARCHITECTURE.md)。

`frame_stop`不包含上界；`data_root`以仓库根目录解析。运行evaluate/identity/trace等模式会写配置指定的输出；使用前检查路径，新结果另存，避免覆盖已有结果。
