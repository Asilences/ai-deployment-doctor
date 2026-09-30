# 同状态提案重放：冻结的开发协议

目标：排除先前 full / no_history 的独立 baseline 与轨迹差异，检验历史可见性和证据表示是否改变提案。**这不是性能实验，不计算新收益，不更新 best，不证明普遍优势或递归自我改进。**

## 四个预先选定的决策状态

配置 [replay-selection-20260930.json](../configs/replay-selection-20260930.json) 固定来源及提案编号，不依据此次输出更换状态：

| ID | 状态 | 原始来源（仅本地） |
| --- | --- | --- |
| S00 | 低并发首项之前，历史为空 | v3.1 serial-search，提案 1 |
| S01 | prefill 首项之前，历史为空 | 两槽位 full J00-search，提案 1 |
| S02 | prefill 拒绝 C02 之后 | 同上，提案 2 |
| S03 | prefill 拒绝 C01 之后，当时隐藏历史 | 两槽位 no_history J01-search，提案 2 |

S03 full 的历史从该次提案之前的真实 trial 重建，未使用第二次候选的结果。选定来源没有“接受后再提案”的状态，明确缺失，不伪造。四状态主要来自一个硬件、一个模型和两个已使用负载；不是四个独立环境，也不是 holdout。

先读取合法 baseline 样本，从过去已接受候选的样本重建 incumbent；只包含严格早于目标提案的历史。记录历史 Git 源码哈希，原始首调用的 payload 哈希必须与重建一致；旧 v3.1 没有完整 evidence 时仍以请求哈希检验。禁止根据 run 的最终 state、目标提案结果或 final 构造输入。best 文件仅计算字节哈希用于只读保护，不读取其中的性能信息。

## 四个输入版本

每状态各一槽位：`original/full`、`original/no_history`、`owned-v1/full`、`owned-v1/no_history`。状态内固定当前配置、精确实测数值、环境、负载、未试列表、模型、系统提示、输出 schema 和生成参数。

`original` 完整重现既有 v3.1 输入表示。`owned-v1` **只用于重放，尚未加入生产搜索 CLI**：

- `current_metrics` 增加测量属于哪个 ID / config / 环境指纹的包装，指标数值原样保留。
- past trial 保留原 gain / reason，补充比较时的 reference ID 与环境指纹；不引入原输入没有的候选绝对指标。
- 所有合法未试候选标为 `measurement_status=unknown`。
- `no_history` 仅将 `prior_trials` 置空；当前测量与相同未试列表仍可见，后者间接透露访问痕迹。

这是一个证据表示组合，也改变 token 长度；不能将差异单独归因于某一个标签或字段。空历史状态内 full / no_history 请求完全相同，作为固定生成参数下的重复调用检查，不作为两个独立样本。

固定 Qwen2.5-1.5B Q4_K_M、llama.cpp b11138、单槽位 / ubatch 128、2048 上下文、temperature 0、seed 42、max_tokens 300、两段解释各不超过 96 字符。目录 ID 和顺序不变；四输入调用顺序用 seed 20260930 预先打乱，每状态成块、串行共享 GPU。

每槽位最多一次格式 / 非法 ID 纠错；失败没有随机兜底。四状态共 **16 槽位、32 次生成调用上限**。传输失败不自动重试；纠错前再次检查输入容量。最多 900 秒，不启动下一槽位，正在执行的请求安全结束。一次规划服务启动的成本单列，不假设各方法总成本相等。

## 容量与记录

固定服务禁用 context shift。每次生成前使用该版本官方 [apply-template / tokenize 接口](https://github.com/ggml-org/llama.cpp/blob/b11138/tools/server/README.md) 计算格式化输入的 token 数，并预留完整 300 输出 token。超限明确失败、不裁切；检查服务实际上下文与返回的 prompt token 数，不匹配则保留失败记录并停止后续任务。

逐调用保存 payload、输入哈希、响应文本、usage、结束原因、解析、错误、时间和纠错。模板与分词检查单独记录，不冒充模型生成调用；缺失用量保留 unknown。运行只读保护检查原始 source / best 文件字节哈希，搜索实验数永远为零。

## 评价与停止条件

主要描述指标：合法提案数、同状态历史开 / 关是否改变 ID、证据归属错误及实际成本。报告所有输入和失败，不能只选有利状态；ID 改变只是信息影响，不证明选择更好。

`review_template.csv` 的标签初始为空。人工按以下公开规则逐字段复核，保留原句和来源：

| 标签 | 判据 |
| --- | --- |
| grounded | 明确引用可见实测，且配置和比较归属正确 |
| prediction | 使用“可能 / 预期”等表达尚未测的效果，没有伪装成实测 |
| unsupported_fact | 将未测性能、最高 / 最优或改善当成已知事实；包括把当前值归到未测候选 |
| unclear | 表述不足以判断上述类别 |

这些标签只审查可见的证据归属和断言形式，不判定完整因果诊断正确性；由研究工程执行者复核，尚无独立标注者一致性证据。统计时说明分母是字段 / 提案 / 调用，分别报告，不能以 LLM 自评代替真实指标。若所有输入都合法但解释仍错误，优先做规划能力诊断；若解释改善，仍需未来独立性能实验。

## 实现与使用

新增 `inference_lab/proposal_replay.py`、`scripts/run_proposal_replay.py` 和针对泄漏、历史轨迹、容量、失败成本、未知用量及原接口一致性的测试。原规划器、控制器、后端与 verifier 不改；默认 v2 与已有 v3/v3.1 接口保持原样。

```powershell
python -B scripts/run_proposal_replay.py --prepare configs/replay-selection-20260930.json
python -B scripts/run_proposal_replay.py --execute runs/<新批次>
python -B scripts/run_proposal_replay.py --summary runs/<新批次>
```

prepare 在任何真实调用前写入冻结状态、输入哈希、顺序、预算和源码身份；execute 要求相同干净提交，拒绝更换状态或实施中改源码。完成槽位不重做，summary 只读。原始运行、snapshot、响应与人工标注仅本地保存；GitHub 公开源码、协议与描述性结果，不包含 `runs/`。

执行后补充：新增只读 `scripts/audit_proposal_replay.py <批次>`，不改上述实验设计、重放实现或原始数据。四状态的完整结果、人工复核与成本见 [实测结果](REPLAY_RESULTS_2026-09-30.md)。
