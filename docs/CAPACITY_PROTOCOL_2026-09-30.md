# 独立规划模型的同状态容量诊断 v1

本轮将规划服务从被优化的推理系统配置中拆出。目标系统仍为 Qwen2.5-1.5B Q4_K_M，原动作空间、门槛、verifier、继承与评分规则不变。本轮仅重放提案，不启动目标系统、不执行候选、不写 best、不产生 final 收益。

## 固定设计

完整复用 [前次重放](REPLAY_PROTOCOL_2026-09-30.md) 的 S00–S03 冻结状态：先审计原批次，再重建并比较 snapshot 字节哈希及决策哈希。每个模型读取相同顺序的四个输入版本：original/full、original/no_history、owned-v1/full、owned-v1/no_history。32 个槽位，各最多一次格式或 ID 纠错，64 次生成调用上限；没有随机回退，传输失败不重试。

规划模型为同系列的 1.5B 和 3B Q4_K_M。所有模型都使用已安装的 llama.cpp b11138 Vulkan、单槽位、ubatch 128、context 2048、batch 512、GPU layers 99、CPU threads 8、temperature 0、seed 42、max_tokens 300、解释各最多 96 字符。规划服务独立端口 8771 / 8772，目标原端口 8765 不启动。

3B 文件的官方固定版本为 `7dabda4d13d513e3e842b20f0d435c732f172cbe`，大小 2,104,932,768 字节，SHA-256 `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d`。来源：[官方模型卡](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)、[固定版本 LFS 校验信息](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/raw/7dabda4d13d513e3e842b20f0d435c732f172cbe/qwen2.5-3b-instruct-q4_k_m.gguf)。模型许可证以官方模型卡为准；本仓库未选定开源许可证。

本机可用显存约 6 GiB、空闲物理内存约 3.65 GiB，因此先选约 2.1 GB 的 3B，而非约 4.7 GB 的 7B。更大不预设为更好。逐模型串行启动服务，按 seed 20260930 冻结模型块顺序；两模型共用同一组状态内调用顺序。只有一次模型块/身份，时序和资源漂移无法消除；模型权重、架构、模板也可能不同，不能把差异完全归因于参数量。

## 独立身份与成本

`local-planning-service-v1` 严格配置只允许服务字段，拒绝 workload、baseline、search_space、门槛或目标 verifier 字段。目标证据始终来自 snapshot；只有实际请求的 model selector 与规划服务 endpoint 依模型配置改变。每次请求明确记录规划 profile、哈希、endpoint、模型路径、完整实际 payload 与哈希，不在传输层偷偷替换模型。

每次调用前按实际规划服务的模板分词，要求输入加 300 输出 token 不超 2048；返回 prompt usage 必须与预检相同。原有失败、未知用量、纠错、结束原因及时间记录保持一致。下载和校验成本另存本地 setup_records，不混入生成成本；实验计时从资产校验后开始，包含两次服务启动、预检、请求、切换及服务结束。900 秒后不启动下一槽位，当前请求安全结束；启动失败不自动降级到其他模型。

旧 inference_lab 与旧重放入口保持字节不变，新代码位于 scripts，保存独立源码指纹。prepare 冻结计划与干净提交；execute 拒绝重复执行或换实现。只读审计先验证真实规划 selector、profile 与 endpoint，再在临时副本上复用旧 schema/输入/成本检查；原始 payload 永不改写。所有旧 source、best 与前次 snapshot/manifest/summary 受字节哈希保护。

## 固定审查规则与边界

沿用此前 grounded / prediction / unsupported_fact / unclear 四类逐字段审查。未测候选“更好 / 最高 / 最优”等无条件断言属于 unsupported_fact；明确未测的可能效果属于 prediction；明确、正确地归属可见实测属于 grounded；不完整或来源不明的表达属于 unclear。记录原句与理由，不能把格式合法当成有依据。研究工程执行者单人复核，无独立标注者一致性证据。

报告每模型全部槽位、合法率、失败、字段分类、历史开关与表示变化、token/耗时/启动成本。四开发状态不独立，空历史的 full/no_history 是重复请求；不进行显著性或普遍优越性宣称。模型提案比较没有目标性能分数，收益未知而非测得零。若 3B 更有依据，下一步仍需冻结候选后独立目标验证；若仍无依据，则下一步比较显式证据引用/预测输出约束，避免直接跳到 Memory 或 L5。

```powershell
python -B scripts/setup_planner.py configs/planner-local-3b.json
python -B scripts/run_capacity_replay.py --prepare configs/capacity-selection-20260930.json
python -B scripts/run_capacity_replay.py --execute runs/<新批次>
python -B scripts/audit_capacity_replay.py runs/<新批次>
```

原始运行、响应、标注与下载记录仅本地保存；以上 runs 路径不是公开数据链接。
