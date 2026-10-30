# 地面点空间与身份评价

更新：2026-10-30。协议源自已验收E004，并用于后续可比实验。是项目自定义地面点IDF1@1m，非官方完整WILDTRACK评测；未实施HOTA。

GT以原personID和positionID转换的米制XY为准，固定半开ROI `[-3,9)×[-9,27)`；不按所用相机可见框改变分母，不因344等疑点删改原GT。每段独立空状态，旧开发0–495/100tick，新开发500–795/60tick，已见参考800–1395/120tick，步长5、dt0.5秒；1400–1995未用。后两段raw组数与ROI内observed数不一定相同。

## 空间

raw_observations、filtered_observed、pure_predictions三个池独立。每时刻每池做欧氏≤1m门内最大合法配对数优先、再最小距离的一对一分配。precision=匹配/域内预测，recall=匹配/GT；mean/median/P90/P95只针对匹配成功子集。纯预测的独立匹配不能加到observed，空间匹配数也不是身份正确数。

full_roi与common_projected分别在各自域内独立匹配。共同区指C1/C2/C3对地面点均正深度且回投在图内，不表示真实无遮挡。E007的区域四集合则按GT位置切分full_roi结果，不冒充历史共同区独立匹配。

## 身份

只用ROI内当前observed滤波XY与track_id，预测点是否in_roi按坐标重算。每tick提供完整GT×预测欧氏距离矩阵，d≤1m合法，其它NaN，不只输入空间获胜关系。motmetrics1.4.0、SciPy求解，显式frameid、max_switch_time=infinity，空帧也更新，缺标注报错。每段/条件独立整段全局对应：

`IDF1=2IDTP/(2IDTP+IDFP+IDFN)=2IDTP/(GT数+预测数)`。

IDP=IDTP/(IDTP+IDFP)，IDR=IDTP/(IDTP+IDFN)。库SWITCH/fragmentation不是逐例看图确认的真实串人次数。近距离或整段IDTP也不保证某帧视觉身份正确，拥挤GT之间可能都在门内；保留这一评价能力边界。

固定实现来源：[motmetrics metrics.py](https://github.com/cheind/py-motmetrics/blob/v1.4.0/motmetrics/metrics.py)及[mot.py](https://github.com/cheind/py-motmetrics/blob/v1.4.0/motmetrics/mot.py)，项目实现见[identity_evaluation](../tracking_demo/identity_evaluation.py)。

## 候选与诊断

E005/E006只更换固定B3组的时间分配，raw成员/XY保持；预登记参考IDF1≥+3pp，两开发各≥−1pp，三段空间P/R各≥−1pp、P90增加≤.05m，任一失败不启用保留段。E005三项身份失败；E006新开发与参考失败，10/12不是“83%改善”。

E007沿用各完整条件保存的整段映射，找合法共现贡献，不分层重求映射；与完整GT集合构造两者均有/仅B2/仅B3/均无。父组按原成员精确恢复，随机与目的样本分开，人工语义/不确定项独立保存。四集合是结果拆分，不是因果贡献分解。见[结果](RESULTS.md)。
