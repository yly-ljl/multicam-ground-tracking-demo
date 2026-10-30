# B3参考段预测数值

更新：2026-10-30。返回[结果导航](../ARCHITECTURE.md)。800–1395，步长5，120tick；三路输入缓存与轨迹来自同一已验收等权B3参考运行，四份1,931,625字节（约1.84MiB），逐字保留。

| 文件 | 数据行 | 字节 |
|---|---|---|
| [C1_observations.csv](C1_observations.csv) | 1778 | 386122 |
| [C2_observations.csv](C2_observations.csv) | 1671 | 362136 |
| [C3_observations.csv](C3_observations.csv) | 2149 | 468984 |
| [tracks.csv](tracks.csv) | 4956 | 714383 |

observations包括检测框/置信度/脚点/投影XY及接受原因，包含被ROI排除的检测；这些是预测，不是GT。源device字段cuda:0记录原检测设备，不表示读取CSV时调用GPU。tracks包括observed与predicted、滤波XY/速度/生命周期/obs_id；track_id只在该段/条件内有效，不是真实personID，不能据总ID数算人数。原缓存的track_id字段可能为空，replay会从组成员重新映射。

使用[public配置](../../configs/public_demo_reference.json)可跳过检测并重做融合/默认跟踪/渲染，仍须自行取得原图和标定，见[快速开始](../../docs/QUICKSTART.md)。4956轨迹状态中observed与纯预测分开，3569融合组不等于独立人数。四CSV不能复算全部七轮研究：没有GT、完整历史组、身份对应/距离或人工图证。

CSV只读，新结果写outputs。来源/哈希见[SOURCE_MANIFEST](../../docs/SOURCE_MANIFEST.csv)，仓库不提供原图、视频、GT或标定副本。
