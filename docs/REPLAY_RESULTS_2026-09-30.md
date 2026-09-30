# 同状态重放结果：选择变化与仍未解决的证据问题

实验前 [冻结协议](REPLAY_PROTOCOL_2026-09-30.md) 与 [四状态选择](../configs/replay-selection-20260930.json) 已发布。真实执行提交为 `28f9b6f8220a5710cc8542c647804390a089a8f1`，Git dirty=false；本地批次 `runs/proposal-replay-20260930-170331-042a37`。

**完成四状态 × 四输入，共 16 次真实模型调用。历史可见性没有改变这些状态上的候选 ID；归属表示改变三个状态的候选，但未消除无依据的性能断言。没有执行新候选或独立性能复测，因而没有新收益分数。**

## 1. 完整候选矩阵

| 固定的真实决策状态 | original/full | original/no_history | owned-v1/full | owned-v1/no_history |
| --- | --- | --- | --- | --- |
| S00：低并发、空历史 | C02 | C02 | C06 | C06 |
| S01：prefill、空历史 | C02 | C02 | C06 | C06 |
| S02：prefill、拒绝 C02 后 | C03 | C03 | C06 | C06 |
| S03：prefill、拒绝 C01 后 | C08 | C08 | C08 | C08 |

C02=(1,256)，C03=(1,512)，C06=(2,256)，C08=(4,64)。当前 incumbent 均 C05=(2,128)；S00 / S01 各 15 个合法未试候选，S02 / S03 各 14 个。禁试由控制器执行，不是模型记忆的证据。

每行四组共享同一精确 state、baseline 指标及可选目录；source 首请求 payload 哈希与重建匹配。S03 原记录隐藏历史，此次 full 仅恢复它此前真实发生的 C01 拒绝，不读取后来 C08 的结果。

S00 / S01 的 full / no_history 在每种表示下输入完全相同，输出文本也完全一致；这四对是固定 seed 的重复调用检查。只有 S02 / S03 有非空历史：两状态 × 两表示，共四个有信息差异的比较，均未改变 ID，但部分解释文本改变。不能把八对或十六输出当成独立环境样本，也不能据此宣称模型永远不能使用历史。

表示改变了 S00、S01、S02 的 ID，S03 不变。owned-v1 同时增加配置归属、环境指纹、unknown 标记和文本长度，因此只能解释为 **这一表示组合影响选择**，不能单独归因于某一个标签，更不能认为 C06 已优于 C02 / C03。

## 2. 输出形式与完整成本

| 版本 | 合法槽位 / 调用 | 纠错 | prompt + completion token | 总 token | 生成耗时（秒） |
| --- | ---: | ---: | --- | ---: | ---: |
| original/full | 4 / 4 | 0 | 3516 + 319 | 3835 | 3.422 |
| original/no_history | 4 / 4 | 0 | 3382 + 331 | 3713 | 3.342 |
| owned-v1/full | 4 / 4 | 0 | 4506 + 266 | 4772 | 2.688 |
| owned-v1/no_history | 4 / 4 | 0 | 4196 + 248 | 4444 | 2.655 |
| 合计 | 16 / 16 | 0 | 15600 + 1164 | 16764 | 12.107 |

全部 `finish_reason=stop`，零无效、重复、缺失 token 用量；全部 prompt token 数与 preflight 一致，最高输入 1187 token，预留 300 输出后未超过固定 2048 上下文。没有裁切历史、随机兜底或目标压测。

规划服务启动 **2.171 秒**；16 次模板 / 分词检查共 **0.486 秒**；整个执行会话 **15.218 秒**。其余约 0.454 秒是记录、调度与关闭等开销，不臆算为某个方法的独占成本。prepare、源文件核对、模型字节哈希、事后人工复核和审计不在执行会话计时内；完整科研成本高于这一数字。

owned-v1 两组共 9216 token，original 两组 7548 token，增加约 **22.10%**。owned-v1 此次生成耗时较短，但共享服务、顺序和长度不同，不能称普遍更快。没有收费 API 调用；GPU 用电与货币成本未知，账本的 `cost=null` 不能解释成零。

## 3. 证据使用仍存在的问题

按照冻结规则，对 32 个字段逐一保留原句与标注说明。标注由项目研究工程执行者复核，未进行独立双人一致性评估；它审查可见的断言与来源，不证明模型内部推理机制。

- **16 个 hypothesis 均包含未限定的最高 / 最佳 / 更高性能断言**，而所选候选未测，按本协议标为 `unsupported_fact`。original 与 owned-v1 各 8 个；绑定输入没有消除此类表述。
- S02 owned/full 的 hypothesis：`C06 has a higher output throughput and better p95_ttft_ms compared to C05.` 输入没有 C06 的测量，不能作为已知比较。
- S03 original/no_history 的 expected_effect：`The model achieves a higher output throughput of 149.6015106636072 tokens/s.` 数字精确来自当前 C05，而 C08 未测，也没有高于当前值；同状态重放再次出现旧错误。
- S03 owned/full 的 expected_effect 使用 `will process 149.6015106636072`。该数仍等于 C05；由于是未来式且句子不完整，标为 `unclear`，不武断称它已经声称测量过 C08。它仍没有说明预测与当前测量的关系。

全部字段的描述性标注为：`unsupported_fact=17`、`prediction=9`、`unclear=6`、`grounded=0`。其中 16 个 unsupported 来自 hypothesis，另一个来自明确的“achieves higher”数值归属；若 expected_effect 是一般未来预期，按 prediction 处理，未来“最佳”且句子不完整的表述保守记为 unclear。标签不等于 32 个独立错误事件，不作为模型总体准确率。

这些结果不支持“归属表示已修复可靠性”。合法 JSON、简短解释和改变候选可以工程验证；真正正确利用失败反馈和性能证据仍未建立，也没有证明历史有害。

## 4. 完整性、测试与本地证据

新增只读 `scripts/audit_proposal_replay.py`，重新核对冻结计划、记录源码身份、原始 source / best 字节哈希、四个 pre-proposal 状态、输入变体、每调用 payload / 生成参数、容量检查、解析、提案、预算和成本汇总。**16 项重放完整性检查通过**；它不评价语义正确性或真实性能。

新增两项篡改检测测试，全部 **75 项测试通过**。现有两个性能开发批次的 7+2 项旧审计也重新通过。新审计与测试在本次模型执行之后增加，重放实现和旧验收器未在实验中改动；九项旧性能审计与十六项提案审计分别计数。

以下全部是本地证据链接，GitHub 不上传 `runs/`：

- [manifest](../runs/proposal-replay-20260930-170331-042a37/manifest.json)、[summary](../runs/proposal-replay-20260930-170331-042a37/summary.json)、[完整性审计](../runs/proposal-replay-20260930-170331-042a37/integrity_audit.json)
- [S00](../runs/proposal-replay-20260930-170331-042a37/S00.json)、[S01](../runs/proposal-replay-20260930-170331-042a37/S01.json)、[S02](../runs/proposal-replay-20260930-170331-042a37/S02.json)、[S03](../runs/proposal-replay-20260930-170331-042a37/S03.json)
- [原表示数值错误 J13](../runs/proposal-replay-20260930-170331-042a37/J13.json)、[归属表示模糊数值 J14](../runs/proposal-replay-20260930-170331-042a37/J14.json)
- [逐字段复核 CSV](../runs/proposal-replay-20260930-170331-042a37/human_review.csv)、[复核说明与哈希](../runs/proposal-replay-20260930-170331-042a37/human_review_summary.json)

```powershell
python -B scripts/audit_proposal_replay.py runs/proposal-replay-20260930-170331-042a37
python -B scripts/run_proposal_replay.py --summary runs/proposal-replay-20260930-170331-042a37
```

## 5. 下一项决策

已完成同状态机制检查与独立输入版本；当前 owned-v1 不直接升级为性能搜索默认版本。下一项优先 **规划模型与被优化模型解耦**，继续保持 Qwen2.5-1.5B 推理目标 / 后端 / verifier 不变，准备更强规划模型在同样 snapshot 上的独立能力对照。OpenRouter Key 暂不可用，较大本地模型需先核对资源；本轮没有安装或运行新模型。

也可另立输出证据引用版本，但必须单独改变 schema 并保留当前结果，不能与换模型合并后宣称单因素收益。只在证据使用显示可信改善后，再投入新任务真实性能对照；继续保留随机、fixed 与全部负结果，Memory 和 L5 仍暂缓。
