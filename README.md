# Inference Improvement Lab

面向大模型推理系统的 **training-free、可验证 L2** 实验工程。目标与验收标准由人固定；Agent 读取真实性能结果，选择下一项推理配置实验；控制器重新启动服务、压测、验证、保留或恢复。L2 采用 [The Last AI Built by Humans §3.3](https://arxiv.org/html/2609.11873v2#S3.SS3) 的“改进策略自主性”定义。

完整研究方向、架构图、文献与优先阅读、当前证据及下一步路线见 [项目总览（2026-09-30）](docs/PROJECT_OVERVIEW_2026-09-30.md)。

新增 [规划器 v3 与成本账本](docs/PLANNER_V3_2026-09-30.md)，通过 `--planner-version v3` 启用；省略时仍使用 v2。实现与实测进度见 [科研工作日志](docs/WORKLOG_2026-09-30.md)。`summary` 命令可只读导出 JSON/CSV。

[v3.1](docs/PLANNER_V31_2026-09-30.md) 增加解释字段长度约束和结束原因账本；[信息消融开发协议](docs/CONTEXT_PILOT_2026-09-30.md) 提供完整信息、环境 / 负载 / 反馈消融与多种子随机、固定规则的独立比较。用 `python -B scripts/run_context_pilot.py --dry-run` 查看冻结计划；这些接口与开发结果不等于环境感知有效性证明。

最新[信息消融实测](docs/CONTEXT_RESULTS_2026-09-30.md)：续跑后七项完成，完整信息版本 final 0%，去掉环境与负载字段 +6.42%，随机 seed 101 +15.99%，seed 100 / 102 均 0%，固定规则 +28.28%。单任务开发结果不支持 Agent 的策略优势。

[两槽位历史反馈实测](docs/HISTORY_RESULTS_2026-09-30.md)：full final 0%，`no_history` +6.98%；首项轨迹不同，未证明历史反馈有益，并发现模型把当前指标附到未测候选。`scripts/audit_context_pilot.py <批次目录>` 提供只读完整性检查；两个批次共九项通过审计，57 项测试通过。

[下一步研究探索与计划](docs/NEXT_STEPS_2026-09-30.md) 核对当前进度并补充相关论文：优先同状态提案重放、测量归属与规划能力诊断，再考虑多任务比较和范围记忆。该文档是计划，本轮未启动新性能实验。

公开仓库保存代码、配置、测试与研究文档。历史文档中的 `../runs/...` 链接指向本地原始证据，GitHub 不包含这些运行目录；模型、vendor 与 API 密钥也不上传。

## 已搭建

- Windows 原生 [llama.cpp](https://github.com/ggml-org/llama.cpp/releases/tag/b11138) Vulkan GPU 后端；[vLLM](https://github.com/vllm-project/vllm/tree/v0.30.0) Linux/WSL2 后端保留作后续扩展。
- 官方 Qwen2.5 GGUF 小模型，固定仓库提交与 SHA-256；Windows 下载脚本验证模型文件。
- 三种实验选择器：同一个本地模型自主提案、实验室 OpenRouter 提案、固定随机搜索对照。
- 两个可改参数：并行推理槽位和物理批处理大小。模型、请求、输入输出长度、验收规则与最大尝试次数固定。
- 每次服务重启后固定预热；参考配置与候选配置交替测量各三次。吞吐提升至少 5%、测量区间不重叠、延迟达标、全部请求输出完整且功能冒烟测试通过才允许保留。
- 保留前重启复核；验证失败则尝试恢复原配置。保留的配置、历史、原始测量和 HTML 报告写入单独的 `runs/` 目录。
- 跨实验继承检查模型、负载、验收规则和后端实现指纹。
- 留出负载审计：冻结已接受的配置，仅改变请求负载，交替重测原始配置与已保存配置；审计不会更新 `best.json`。
- 探索性多任务对照：固定规则、随机搜索和本地 Agent 使用独立起点；搜索后重新测量最终保留配置。

## Windows 本机运行

```powershell
cd D:\Projects\ai-deployment-doctor
python scripts/setup_windows.py --config configs/windows-1.5b.json
python -m inference_lab doctor
python -m inference_lab baseline --config configs/windows-1.5b.json
python -m inference_lab run --config configs/windows-1.5b.json --planner local --iterations 1
python -m inference_lab run --config configs/windows-1.5b.json --planner random --iterations 1
python -m inference_lab audit --config configs/holdout-windows-1.5b.json --candidate runs/llama_cpp-local-20260924-022457-406ab4/best.json
python -m inference_lab run --config configs/pilot-serial-windows-1.5b.json --planner fixed --iterations 1
```

已安装的 llama.cpp Vulkan 程序和模型可重复使用。每次命令自动新建报告目录；打开最新的 `runs/llama_cpp-*/report.html`。完整安装及可选 WSL2 路线见 [启动说明](docs/SETUP.md)。

0.5B 模型已完成真实 GPU 基线和随机搜索实验；它两次原样重复当前配置，说明其自主提案能力不足。1.5B 本地模型在一轮真实实验中自主提出 `(parallel=4, ubatch_size=512)`，吞吐中位数从约 247 提高到 321 输出 token/秒，独立验收及重启复核通过。新运行继承改进配置后复测中位数约 321 输出 token/秒；下一轮本地模型提出的候选反而下降约 24.4%，因此被拒绝，改进配置得到保留。详细条件和局限见 [实验结果](docs/RESULTS_2026-09-24.md)。本地模型提案无需 API Key 或权重训练，但仍消耗推理算力。

冻结首轮已保存的配置后，两组不同请求负载的只读审计分别测得 +22.8% 和 +19.3% 的吞吐差异；详见 [留出负载审计](docs/HOLDOUT_2026-09-24.md)。这只检验了负载形状变化，尚未验证不同任务、模型或机器上的效果。

同预算的三次 Agent 与三次随机搜索试验中，Agent 三次都选中有效配置，随机搜索一次抽中相同配置并通过验收；详见[探索性对照](docs/COMPARISON_2026-09-24.md)。这不足以证明 Agent 普遍优于随机搜索。

为检验环境感知，新增低并发和高并发长输入两组[探索性试点](docs/PILOT_RESULTS_2026-09-24.md)。低并发时 Agent、随机搜索和固定规则都没有可验收提升；高并发时 Agent 搜索后独立收益 +24.4%，随机种子 100 为 +9.8%，但固定规则同样达到 +24.9%。v1 Agent 还重复已拒绝配置或产生无效提案。随后加入禁测历史和一次纠错的[规划器 v2](docs/PLANNER_V2_2026-09-24.md)，避免重复执行，但尚未在低并发任务找到有效新候选。当前结果不能证明优于随机或固定规则。

## 研究结论的边界

优化对象是由模型、服务及配置组成的推理系统。模型权重不改变；一次配置增益不代表模型本身变聪明，也不证明开放式 RSI。随机搜索得到的性能提升只能证明测量与验收闭环，本地或 OpenRouter Agent 的 L2 证据需要其自主提案经真实实验保留并在后续轮次继承。

当前默认负载仅 16 个请求，适合工程演示；正式比较需要扩大请求量、独立重复、相同实验预算和未参与选择的工作负载。三个固定 prompt 的一致性检查只是功能冒烟测试，不是完整语义质量评测。5% 与测量区间不重叠是保守工程门槛，不等同统计显著性。

`.env`、模型文件、下载程序和 `runs/` 均不会提交到版本控制。OpenRouter 密钥目前未配置；拿到实验室密钥后按 [启动说明](docs/SETUP.md) 本地填写即可切换 `--planner openrouter`，无需改动验收器。
