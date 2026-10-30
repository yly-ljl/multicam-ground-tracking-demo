# 精确汇总表

更新：2026-10-30。返回[结果导航](../ARCHITECTURE.md)。均从原表逐字复制，不重新四舍五入或优化指标。字段/比较口径见[评价协议](../../docs/EVALUATION.md)，解释见[RESULTS](../../docs/RESULTS.md)。

| 文件 | 内容 |
|---|---|
| [E002_spatial.csv](E002_spatial.csv) | 旧开发统一ROI三池空间表 |
| [E003_development_spatial.csv](E003_development_spatial.csv) | 等权/加权开发空间 |
| [E003_reference_spatial.csv](E003_reference_spatial.csv) | 等权/加权已见参考空间 |
| [E004_identity.csv](E004_identity.csv) | 三段C1/B2/B3整段身份 |
| [E004_spatial.csv](E004_spatial.csv) | 三段/条件空间汇总 |
| [E005_identity.csv](E005_identity.csv) | B3与GLOBAL身份 |
| [E005_thresholds.csv](E005_thresholds.csv) | E005十二项门槛 |
| [E006_identity.csv](E006_identity.csv) | B3/SOFT及历史GLOBAL明确来源 |
| [E006_thresholds.csv](E006_thresholds.csv) | E006十二项门槛 |
| [E007_four_sets.csv](E007_four_sets.csv) | 三套四集合及GT区域切分 |
| [E007_manual_counts.csv](E007_manual_counts.csv) | 固定有限样本分层支持/不确定计数，非总体比例 |
| [E007_structure.csv](E007_structure.csv) | 三段三结构分母与机械结果 |

raw、filtered_observed和pure_predictions不可相加；各段分母独立。E007按GT位置切分full_roi不等于历史共同区独立匹配。source列的旧outputs路径仅是来源标识，不是发布树可打开链接。
