# 独立规划模型真实性能开发试验 v1

回答：在相同目标模型、动作空间与外部验收器下，独立 3B 规划服务的提案能否获得真实可验收收益，是否值得继续投入？不要求正结果，也不声称普遍优于随机搜索。

## 实验前固定内容

使用已有两个开发负载（非 holdout）：serial 为 16 请求、64 输入 / 32 输出 token、并发 1、seed 1001；prefill 为 24 请求、384 输入 / 32 输出 token、并发 8、seed 1002。目标始终为 Qwen2.5-1.5B Q4_K_M、llama.cpp b11138 Vulkan，目标端口 8765。

每任务比较独立 1.5B 规划、独立 3B 规划、随机搜索 seeds 100/101/102、固定 C11 `(4,512)`，共 12 项。冻结 original/full 输入；两规划模型同系统提示、schema、temperature 0、seed 42、300 输出 token、96 字符解释，单规划槽位 / ubatch 128 / ctx 2048。不同时调整表示或加入证据引用约束，避免混合干预。配置与顺序见 [选择文件](../configs/planner-performance-20260930.json)。

每项独立测新 baseline，初始 C05 `(2,128)`，不继承旧 best，不使用旧成绩代替新对照。每方法候选槽位上限 2，固定预设只使用一次，明确不是等实际成本；两个模型每槽位最多一次格式 / ID 纠错，无随机回退。全批实际候选执行上限 22，模型生成调用上限 16。随机种子在任务内报告，不充当三个独立环境。

目标服务在每次规划前停止，独立规划服务完成请求并结束后，再启动目标做性能验证；共享 GPU 串行使用，不能让 3B 占用显存影响目标成绩。逐次保存实际 planner profile / endpoint / model selector、完整请求、预检与 ledger。动作仅由 candidate_id 映射；解释错误单独保留，不假设语言正确才允许合法候选测量。

服务启动、预热、三个功能 prompt、交替 AB 各三次、至少 5% 且区间不重叠、延迟 / 完成 / token 门槛、保留前重启复核与拒绝后恢复全部沿用旧 verifier。最终选定后，再做独立 final 交替测量，不向规划器返回 final；若仍保留 baseline，复测 baseline 并按原评分定义记收益 0%，不是认为未测候选的效果为零。

## 版本身份、成本与时间块

新控制器仅在 scripts 中实现，旧 inference_lab 和旧实验入口不变。所有方法的 settings 加入相同 `experiment_controller_sha256` / protocol 元数据，使新 best 的系统身份包含新控制器源码；旧 settings 不能绕过继承检查，新 final 使用相同增强身份。实际模型 / 负载 / 门槛未变，新旧成绩不混作同协议对照。独立规划服务不是修改模型权重。

prepare 在任何新性能调用前校验两个模型资产、冻结目标 settings、profile、顺序、预算、干净提交与源码指纹，并对历史 run / best / final / manifest 及输入文件做只读哈希保护。随机化任务块与块内方法顺序，所有 GPU 操作串行。

按最多 900 秒的可恢复时间块执行。剩余不足 360 秒不启动新的完整搜索，该阈值来自此前约数分钟的每项成本，只是工程保守估计。每个搜索收到剩余软预算，超时后不启动下一槽位，允许已经开始的 AB / 恢复安全结束；final 若尚未开始则保存 search_completed，下块接着完成。已开始 final 安全结束，不强杀目标服务。失败或中断搜索不重跑、不暗中补预算；缺少 final 保持未知。

新运行完整保留调用成本，并另外记录规划启动、预检、关闭、规划总耗时及目标 start / functional / warmup / measure / stop 的操作事件；嵌套 stop 已包含在 start，汇总只加顶层事件，避免重复计算。搜索壁钟与独立 final 壁钟并列报告，不把各阶段重叠时间相加当总成本。未返回 token 为未知，电费 / 货币未知，不能默认为零。下载成本沿用单独资产准备记录，不混入生成成本。

## 分析与边界

报告所有 12 项的 proposal ID、接受 / 恢复、最终配置、独立 final 吞吐及延迟、合法 / 无效 / 纠错、token、搜索 / final / 总任务成本。每任务列模型与 fixed / 三随机种子的差值；失败、pending、baseline 保留与有效零收益分别呈现。只做描述性任务级比较，不以重复样本扩充显著性。

旧 1.5B 单机性能收益仍是原先证据，这轮不覆写。两个开发负载、单硬件、单模型、较少请求和一次方法顺序不能说明跨环境、统计显著性或普遍优势。3B 更大不预设更好，合法提案或预测措辞改善不能代替目标测量。

只读审计核对独立模型身份、真实 live snapshot 与严格过去历史、合法 ID、原始 AB gate / 重启 / 恢复、最终 source / best / settings 哈希、成本及功能输出。新原始 runs 仅本地；GitHub 上传代码、测试、协议、结果和工作日志，不上传原始数据、模型或凭据。

```powershell
python -B scripts/run_planner_performance.py --prepare configs/planner-performance-20260930.json
python -B scripts/run_planner_performance.py --execute runs/<批次> --max-seconds 900
# partial 时，同源码 / 同提交继续下一个时间块；不重复已消耗项
python -B scripts/audit_planner_performance.py runs/<完整批次>
```

取得全部结果后再决定证据引用、动作一致性输出或新负载留出比较的优先级；本轮暂不改原输出规则、引入 Memory / Probe / L5。
