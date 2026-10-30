# 精确汇总表

更新：2026-10-30。返回[结果导航](../ARCHITECTURE.md)。表中保留原始数值精度和字段名。GT为真实标注，C1为单相机，B2为双相机C1+C2，B3为三相机C1+C2+C3；GLOBAL/SOFT为两个未采用的时间关联候选。字段/比较口径见[评价协议](../../docs/EVALUATION.md)，解释见[RESULTS](../../docs/RESULTS.md)。

| 文件 | 内容 |
|---|---|
| [E002_spatial.csv](E002_spatial.csv) | 旧开发的整个地面评价区域（ROI）内，跟踪前观测/有观测滤波轨迹/纯外推轨迹三个池的位置评价 |
| [E003_development_spatial.csv](E003_development_spatial.csv) | 等权/加权开发空间 |
| [E003_reference_spatial.csv](E003_reference_spatial.csv) | 等权/加权已使用的对照片段空间 |
| [E004_identity.csv](E004_identity.csv) | 三段C1/B2/B3整段身份 |
| [E004_spatial.csv](E004_spatial.csv) | 三段/条件空间汇总 |
| [E005_identity.csv](E005_identity.csv) | B3与GLOBAL身份 |
| [E005_thresholds.csv](E005_thresholds.csv) | E005十二项门槛 |
| [E006_identity.csv](E006_identity.csv) | B3/SOFT及历史GLOBAL明确来源 |
| [E006_thresholds.csv](E006_thresholds.csv) | E006十二项门槛 |
| [E007_four_sets.csv](E007_four_sets.csv) | 分别按跟踪前位置、滤波位置、整段身份匹配，把GT分成双/三相机均有、仅双有、仅三有、均无；另按GT位置划分区域 |
| [E007_manual_counts.csv](E007_manual_counts.csv) | 固定有限样本分层支持/不确定计数，非总体比例 |
| [E007_structure.csv](E007_structure.csv) | 三个片段中C3独立新组、C3加入已有C1/C2组、无C3成员组三种结构的数量与匹配统计 |

旧/新开发表示同场景片段的使用顺序；`seen_reference`对应已使用的对照片段（800–1395），原在E003用于验证、后用于分析比较，不是未使用测试段。`raw_observations`为跟踪前观测，`filtered_observed`为当前有观测支持的滤波轨迹，`pure_predictions`为当前只有外推的轨迹；三者分别评价，不可相加，各段分母独立。E007按GT位置切分full_roi不等于历史共同区独立匹配。source列的旧outputs路径仅是来源标识，不是发布树可打开链接。
