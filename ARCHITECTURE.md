# 仓库目录导航

更新：2026-10-30。本页描述发布树，系统关系见[数据流](docs/ARCHITECTURE.md)。

| 入口 | 职责 |
|---|---|
| [run_demo.py](run_demo.py) | demo播放/缓存回放/图像推理，以及原研究模式入口；默认配置仍是原C1，入门请显式用public_demo_reference。 |
| [tracking_demo](tracking_demo/ARCHITECTURE.md) | 预测、融合、跟踪、独立评价、候选与E007诊断全部模块，原注释/说明字符串/排版保留。 |
| [configs](configs/ARCHITECTURE.md) | 原研究配置原样保留；public_demo_reference单独连接发布CSV与读者的外部资产。 |
| [tests](tests/ARCHITECTURE.md) | 原48项CPU测试，其中2项需要外部标定。 |
| [scripts](scripts/ARCHITECTURE.md) | 入门测试封装及原研究辅助脚本，逐项写明前置档案与副作用。 |
| [docs](docs/ARCHITECTURE.md) | 快速开始、代码阅读、结果、协议、限制、来源和自绘图。 |
| [experiments](experiments/README.md) | E001–E007简短对外报告与索引，非内部逐步审批档案。 |
| [results](results/ARCHITECTURE.md) | 精确汇总数值和四份B3参考预测CSV，不含GT文件或原图。 |
| [requirements.txt](requirements.txt) | 已用Windows/Python3.12.6的固定依赖；包含Windows CUDA wheel直链。 |
| [LICENSE](LICENSE)、[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md) | 项目源码许可及第三方/模型/数据边界。 |

data/、weights/由读者自行放置；outputs/、.cache/、.venv/为本地运行产物，不提交。新结果另存新目录，禁止覆盖参考数值或历史证据。复杂目录就近维护导航；简单报告、同构条件子树由上级说明，不机械增加文档。源码/注释修改遵循[AGENTS](AGENTS.md)。

## 阅读文档入口

| 文档 | 作用 |
|---|---|
| [QUICKSTART](docs/QUICKSTART.md) | 环境、资产、images/replay/play和测试前提。 |
| [CODE_READING](docs/CODE_READING.md) | 函数级阅读路线和变量/片段定位。 |
| [RESULTS](docs/RESULTS.md) | E001–E007完整数字、收益、代价与不采用结论。 |
| [EVALUATION](docs/EVALUATION.md) | ROI、分母、空间与整段身份协议。 |
| [LIMITATIONS](docs/LIMITATIONS.md) | 已知失败、因果和泛化限制。 |
| [DATA_AND_MODELS](docs/DATA_AND_MODELS.md) | 官方资产来源、放置方式及许可边界。 |
| [PROVENANCE](docs/PROVENANCE.md)、[VERSION](docs/VERSION.md) | 原研究到发布副本的来源和新历史。 |
| docs/assets/ | 由汇总数字/代码关系自绘的SVG，不含数据集画面。 |
| docs/third_party/、[DEPENDENCIES.csv](docs/DEPENDENCIES.csv) | 本地已安装依赖的许可文本/元数据摘要；没有分发wheel或第三方源码。 |
