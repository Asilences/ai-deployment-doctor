# 实验室 API 准备

截图中“未配置 / 尚未保存 API Key”仅说明该页面没有保存密钥，不能证明实验室尚未开通额度。

用户提供的桌面 OpenRouter.py 使用 GET /api/v1/credits 查询账户额度；Authorization 是占位内容。它不是模型调用示例，也没有提供可用密钥或模型 ID。

请向实验室管理员确认：
1. 获取本人实验用的 OpenRouter 推理 API Key，或加入实验室组织后创建实验用 Key。
2. 是否直接使用 https://openrouter.ai/api/v1，还是必须经实验室代理。
3. 获准使用的模型 ID，以及限流和额度规则。

不要申请管理密钥；推理调用使用普通 API Key。
密钥配置在本项目 .env 中；该文件已被 .gitignore 排除。
无需把密钥粘贴到聊天中。准备阶段未调用收费接口。
