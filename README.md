# AutoEmail

# LangGraph 邮件自动化回复系统

基于 LangGraph 构建的智能邮件客服系统，支持自动拉取待处理邮件、按类别路由、结合 RAG 与长期记忆生成客服回复草稿，并通过审校与重写机制控制输出质量。

## 项目定位

- **适用场景**：SaaS 客户支持、客服邮件分流、高一致性自动回复
- **运行方式**：逐个处理收件箱中的待回复邮件，生成邮件草稿供人工确认后发送
- **设计原则**：在回复质量可控的前提下，减少人工重复工作，保持语气、策略与信息安全策略一致

## 核心特性

- 多智能体工作流：分类智能体、RAG 检索智能体、撰写智能体、审校智能体协同完成邮件处理
- 邮件分类与路由：自动识别产品咨询、投诉反馈、无关邮件，并执行差异化处理策略
- RAG 知识增强：基于 Chroma 向量库检索内部知识，为产品咨询类邮件提供可溯源依据
- 长期记忆：按发件人维护历史互动摘要与回复策略，实现个性化回复
- 上下文预算管理：整合顶层规则、长期记忆、RAG 结果和短期对话历史，控制上下文长度
- 审校与重写：审校不通过时自动重新组装上下文并重写回复，最多支持 3 轮迭代

## 技术栈

- 工作流编排：LangGraph
- 大语言模型：DeepSeek、智谱 AI Embedding
- 向量检索：Chroma + ZhipuAIEmbeddings
- 数据存储：SQLite
- 邮件接入：Gmail API、QQ 邮箱、网易邮箱
- 运行环境：Python 3.10+，异步执行

## 项目结构

```
langgraph-email-automation/
├── main.py                     # 工作流启动入口
├── requirements.txt            # 依赖清单
├── context/
│   └── company_rules.md        # 客服回复顶层规则
├── src/
│   ├── state.py                # LangGraph 状态定义
│   ├── graph.py                # 工作流图与节点路由
│   ├── agents.py               # LLM 智能体链定义
│   ├── nodes.py                # 节点业务逻辑
│   ├── prompts.py              # 提示词模板
│   ├── structure_outputs.py    # 结构化输出定义
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── GmailTools.py       # Gmail API 封装
│   │   ├── QQEmailTools.py     # QQ 邮箱工具封装
│   │   ├── QQMailTools.py      # QQ 邮箱兼容封装
│   │   └── NeteaseEmailTools.py # 网易邮箱工具封装
│   ├── context/
│   │   ├── __init__.py
│   │   └── context_manager.py  # 多层上下文组装与预算控制
│   └── memory/
│       ├── __init__.py
│       └── sender_memory.py    # 发件人长期记忆与检索
├── db/                         # 本地数据库与向量库
└── workflow.png                # 工作流示意图
```

## 工作流概览

1. 从邮箱读取最近未回复邮件
2. 判断待处理队列是否为空
3. 对当前邮件进行分类
4. 对产品咨询类邮件构造检索查询并从内部知识库检索
5. 组装顶层规则、长期记忆、RAG 结果和短期对话历史
6. 撰写邮件回复草稿
7. 审校回复质量
8. 审校通过则保存草稿；若未通过则在预算范围内重写，最多 3 轮；无关邮件直接跳过

## 环境配置

在项目根目录创建 `.env` 文件，至少包含以下变量：

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key
ZHIPU_API_KEY=your_zhipu_api_key
USE_GMAIL=true
EMAIL_PROVIDER=qq
MY_EMAIL=your_email@example.com
```

首次使用 Gmail 时，程序会触发 OAuth 授权，并生成本地授权文件，请勿提交到代码仓库。

## 公司回复规则

系统会读取 `context/company_rules.md` 作为全局约束，核心要求包括：

- 保持礼貌、简洁、专业
- 不得承诺未经内部确认的功能、退款、定价例外或 SLA 结果
- 信息不足时明确说明，并引导用户到正确支持路径
- 不得暴露内部工具、系统细节或原始推理
- 对疑似钓鱼、滥用或安全事件，优先输出安全的非破坏性回复

## 快速开始

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd langgraph-email-automation

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 并填入真实密钥

# 5. 运行工作流
python main.py
```

程序会按顺序处理收件箱中的新邮件，并在每封邮件处理完成后输出节点执行日志。

## 可扩展方向

- 支持审校通过后直接发送邮件
- 扩展 HTML 富媒体模板与附件支持
- 抽象邮件服务层，接入更多邮箱服务商
- 支持向量库在线更新
- 扩展为多租户或团队隔离的记忆体系
