# 代码阅读路线

更新：2026-10-30。按函数和起止片段定位，不依赖注释增减后的固定行号。发布副本原49份代码/配置/测试/依赖逐字保留，源码中的中文学习说明不是整理时删改的对象。

| 顺序 | 入口与片段 | 应联结的问题 |
|---|---|---|
| 1 | [run_demo.main](../run_demo.py)，`parser.add_argument('mode'...)`到`execute(...)`分支 | images/replay/play怎样选择？默认配置为什么仍需显式替换？ |
| 2 | [demo_delivery.execute / fresh_output](../tracking_demo/demo_delivery.py)，输出校验到mode分派 | 新目录保护怎样避免重写旧结果？ |
| 3 | 同文件`sequence`，`cameras={...}`到`frame_ids=...` | 标定、tracker、renderer、writer各维护什么？ |
| 4 | 同文件`if model is None`到`local.append(...)` | 缓存观测与YOLO结果怎样形成相同的观测字段？底中心不是独立足点。 |
| 5 | [geometry.GroundCamera.__init__ / to_ground / to_image](../tracking_demo/geometry.py) | 世界→相机R/t、t_cm/100、H和H_inv怎样对应？depth/ROI为什么分别检查？ |
| 6 | [fusion.fuse_observations](../tracking_demo/fusion.py)，相机循环→allowed/costs→matched→组均值 | 完整成员门与点到组均值的代价不同；虚拟项与单组如何保留未匹配观测？ |
| 7 | [tracking.MetricFilterFactory / GroundTracker.update](../tracking_demo/tracking.py) | [X,Y,Vx,Vy]的F/Q、真实0.5秒、补空tick与Norfair生命周期如何分开？ |
| 8 | 回到`sequence`，`mapping={...}`到`renderer.draw(...)` | 原obs_id如何经fusion_id映到track_id，observed与predicted如何展示？ |
| 9 | [showcase.ShowcaseRenderer](../tracking_demo/showcase.py) | 总览保留全部状态，zoom只是范围，不是筛掉错误；轨迹历史按时间裁剪。 |
| 10 | [analysis.gated_pairs / compare_multicam](../tracking_demo/analysis.py) | 空间1m门内一对一评价，匹配距离为何是条件子集？ |
| 11 | [identity_evaluation.distance_matrix / accumulate / identity_mapping](../tracking_demo/identity_evaluation.py) | 完整距离关系与整段对应为何不能用帧内获胜对或CLEAR事件替代？ |
| 12 | [association](../tracking_demo/association.py)、[observation_diagnostics](../tracking_demo/observation_diagnostics.py) | 未采用的时间目标与只读诊断在什么边界外运行？ |

## 三种ID与三类“门”

`C3_00001050_003`之类obs_id指特定相机/时刻的检测；`B3_00001050_020`指一帧的融合组；track_id才跨时刻维护，且只在本段/条件内有效。融合0.75m约束成员互相距离，时间关联严格<1.5m，事后身份/空间评价则≤1m，三者不可混用。

## 从测试看反例

[test_fusion](../tests/test_fusion.py)的transitive-chain用例说明不能仅靠链式近邻合组；[test_core](../tests/test_core.py)的1秒间隙说明离散period不是物理dt；[test_identity](../tests/test_identity.py)的full_relations用例说明空间最近者交替会改变错误输入下的IDF1；[test_soft_assignment](../tests/test_soft_assignment.py)的255/840局部算例不代替三段采用门槛；[test_observation_diagnostics](../tests/test_observation_diagnostics.py)的condition roles说明两条件拼图并集不是单个B3组。

研究配置/诊断脚本的文件前提见各目录导航。阅读函数不等于应立即执行历史脚本；若涉及实际数据写出，先确认配置来源、输出新目录及可比条件。
