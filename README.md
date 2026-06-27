# AutoEmailResponse

基于 LangGraph 的邮件智能客服系统，支持 **QQ 邮箱** 与 **163 邮箱**。自动拉取待回复邮件，经多智能体流水线生成回复，审校通过后写入邮箱草稿箱，供人工确认发送。

## 工作流程

```
收件箱(IMAP) → 分类 → [产品咨询: RAG检索] → 组装上下文 → 撰写 → 审校 ⇄ 重写(≤3次) → 草稿箱(IMAP)
                              ↓ 无关邮件 → 跳过
```

| 邮件类型 | 处理方式 |
|---------|---------|
| 产品咨询 | RAG 检索知识库 + 生成回复 |
| 投诉 / 反馈 | 直接组装上下文生成回复 |
| 无关邮件 | 跳过 |

## 技术栈

LangGraph · DeepSeek · 智谱 Embedding · Chroma · SQLite · Python 3.10+

## 目录结构

```
AutoEmailResponse/
├── main.py                      # 启动入口
├── requirements.txt
├── context/                     # 知识库源文件
│   └── company_rules.md         # 客服回复规则
├── src/
│   ├── graph.py                 # LangGraph 工作流
│   ├── nodes.py                 # 节点逻辑
│   ├── agents.py                # LLM 智能体
│   ├── prompts.py               # Prompt 模板
│   ├── tools/
│   │   ├── QQMailTools.py       # QQ 邮箱 IMAP/SMTP
│   │   ├── NeteaseEmailTools.py # 163 邮箱 IMAP/SMTP
│   │   └── imap_common.py       # IMAP 草稿写入
│   ├── context/                 # 多层上下文组装
│   ├── memory/                  # 发件人长期记忆
│   ├── observability/           # Trace 可观测性
│   └── Rag/                     # 向量索引与检索
├── eval/                        # RAG 评测（检索 / faithfulness）
└── db/                          # Chroma 向量库 + SQLite（本地生成，不入库）
```

## 快速开始

```bash
git clone https://github.com/hairuipile/AutoEmailResponse.git
cd AutoEmailResponse

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
cp .env.example .env            # 填入密钥后运行
python main.py
```

首次运行会自动检测 `context/` 知识库变更并更新向量索引。

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `EMAIL_PROVIDER` | 否 | `qq`（默认）或 `163` |
| `MY_EMAIL` | 是 | 邮箱地址 |
| `QQ_EMAIL_AUTH_CODE` | qq 时 | QQ 邮箱 IMAP/SMTP 授权码 |
| `NETEASE_EMAIL_AUTH_CODE` | 163 时 | 163 邮箱授权码 |
| `DEEPSEEK_API_KEY` | 是 | DeepSeek 对话 API |
| `ZHIPUAI_API_KEY` | 是 | 智谱 Embedding API |
| `LLM_PROVIDER` | 否 | `DEEPSEEK`（默认）或 `ZHIPUAI` |
| `TRACE_ENABLED` | 否 | 是否开启 Trace，默认 `true` |
| `TRACE_LOG_PATH` | 否 | Trace 日志路径，默认 `logs/traces.jsonl` |

### QQ 邮箱

1. [QQ 邮箱](https://mail.qq.com) → 设置 → 账户 → 开启 IMAP/SMTP
2. 生成授权码 → 填入 `QQ_EMAIL_AUTH_CODE`

```env
EMAIL_PROVIDER=qq
MY_EMAIL=your@qq.com
QQ_EMAIL_AUTH_CODE=xxxx
```

### 163 邮箱

1. [163 邮箱](https://mail.163.com) → 设置 → POP3/SMTP/IMAP → 开启服务
2. 生成授权码 → 填入 `NETEASE_EMAIL_AUTH_CODE`

```env
EMAIL_PROVIDER=163
MY_EMAIL=your@163.com
NETEASE_EMAIL_AUTH_CODE=xxxx
```

## 核心模块

- **分类智能体**：识别产品咨询、投诉、反馈、无关邮件
- **RAG 检索**：Chroma 向量库检索 `context/` 知识，两步式检索+生成
- **上下文管理**：顶层规则 + 长期记忆 + RAG 结果 + 短期历史，带 token 预算裁剪
- **撰写 + 审校**：结构化输出，审校不通过自动重写，最多 3 轮
- **邮件接入**：IMAP 拉取未回复邮件，IMAP APPEND 写入草稿箱

## 自定义

编辑 `context/company_rules.md` 配置全局回复约束（语气、禁止承诺事项、信息安全等）。

## 评测

```bash
python eval/build_dataset.py --limit 30          # 合成评估集
python eval/retrieval_eval.py                    # Recall@K / MRR
python eval/ragas_eval.py --limit 5              # faithfulness（消耗 API）
```
