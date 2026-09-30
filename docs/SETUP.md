# 启动说明

## Windows 原生路线（当前推荐）

本机 GPU：RTX 3070 Ti Laptop，8 GB 显存。使用 llama.cpp 的 Windows Vulkan 官方发布包，无需 Docker、WSL 或管理员权限。下载脚本固定发布标签 b11138 和 Qwen 模型提交，按 Hugging Face 官方 LFS SHA-256 校验模型文件。

```powershell
cd D:\Projects\ai-deployment-doctor
python scripts/setup_windows.py --config configs/windows-1.5b.json
python -m inference_lab doctor
python -m unittest discover -s tests -v
python -m inference_lab baseline --config configs/windows-1.5b.json
python -m inference_lab run --config configs/windows-1.5b.json --planner local --iterations 1
```

`configs/windows.json` 是更轻的 0.5B 模型，适合检查服务和测量流程；它在现有实验中连续两次提出原配置，不能把这两轮称为成功的 L2 改进。默认配置现为 `configs/windows-1.5b.json`，其首轮结果见 [实验结果](RESULTS_2026-09-24.md)。选择配置后，基线、随机搜索、Agent 运行必须使用同一配置文件。程序只绑定 127.0.0.1:8765；若端口被占用会拒绝启动，不会接管既有服务。

每次运行把报告和日志保存到 `runs/`。`report.html` 展示提案及验收结果，`run.json` 和每轮 `benchmark.json` 保存细节，`best.json` 保存经过复核的状态。可用 `--inherit runs/某次实验/best.json` 开始下一轮；若模型、负载或验收器版本不一致，继承会被拒绝。

已保存配置也可以在未参与选择的负载上进行只读审计。审计要求候选来自已完成的成功实验，除 `workload` 外配置必须与来源实验相同，不会继承或更新 `best.json`：

```powershell
python -m inference_lab audit --config configs/holdout-windows-1.5b.json --candidate runs/llama_cpp-local-20260924-022457-406ab4/best.json
python -m inference_lab audit --config configs/holdout-long-windows-1.5b.json --candidate runs/llama_cpp-local-20260924-022457-406ab4/best.json
```

详细数值和局限见[留出负载审计](HOLDOUT_2026-09-24.md)。

随机搜索可用 `--seed 731` 等非负整数固定候选抽样；种子会写入新运行的提案记录。对照试验应使 Agent 和随机搜索使用相同的 `--config`、`--iterations` 和独立起点。已完成的三轮小样本记录见[探索性对照](COMPARISON_2026-09-24.md)。

搜索后的独立最终评分使用搜索任务的**原配置文件**和该次运行的 `best.json`，要求模型、负载和评测器源码指纹一致。若最终仍是原始配置，只重新测量这一配置并记为 0% 相对收益：

```powershell
python -m inference_lab run --config configs/pilot-serial-windows-1.5b.json --planner local --iterations 2
python -m inference_lab run --config configs/pilot-serial-windows-1.5b.json --planner random --seed 100 --iterations 2
python -m inference_lab run --config configs/pilot-serial-windows-1.5b.json --planner fixed --iterations 1
python -m inference_lab final --config configs/pilot-serial-windows-1.5b.json --candidate runs/某次实验/best.json
```

请把最后一行的路径改为刚完成运行的 `best.json`。已完成的探索性试点见[协议](PILOT_PROTOCOL_2026-09-24.md)与[结果](PILOT_RESULTS_2026-09-24.md)。

使用 OpenRouter 时，只在项目根目录的 `.env` 本地填写推理 Key 与允许使用的模型 ID：

```powershell
Copy-Item .env.example .env
python tools/check_openrouter.py
python tools/check_openrouter.py --completion
python -m inference_lab run --config configs/windows-1.5b.json --planner openrouter --iterations 1
```

目前没有实验室 Key，因此上述请求尚未执行。不要把密钥粘贴到聊天或提交到 Git。OpenRouter 只接收结构化实验数据；测量服务器子进程不继承远程 API 密钥。

## Linux / WSL2 vLLM 路线（后续）

vLLM 官方不原生支持 Windows。本机 WSL 服务当前禁用，需管理员执行 `scripts/enable_wsl_service.ps1`，可能还要安装 Ubuntu 并重启。完成后可在 Ubuntu 运行：

```bash
cd /mnt/d/Projects/ai-deployment-doctor
bash scripts/setup_linux.sh
source ~/.venvs/inference-lab/bin/activate
python -m inference_lab baseline --config configs/local.json
python -m inference_lab run --config configs/local.json --planner openrouter --iterations 1
```

`configs/local.json` 固定 vLLM 0.30.0 和 Qwen2.5-0.5B-Instruct 的 Hugging Face 提交；真实 Linux 适配器尚未在本机验证。Linux 与 Windows 的模型格式、运行时、可改参数不同，结果不能直接互作对照。

官方文档：[llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)、[vLLM 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)、[WSL 安装](https://learn.microsoft.com/en-us/windows/wsl/install)。
