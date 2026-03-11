# Miro SciFi

一个基于 Python + LangGraph 的社会派科幻多智能体推演原型，已经从“能跑通”升级为更贴近严肃文学生产的 Agent 系统：

- 用 `Showrunner` 在场景开始前生成节拍表，先锁定文学目的，再让角色进入冲突。
- 给角色加入数值化资源池，让生存压力而不是温情本能驱动行为。
- 把记忆拆成 `short_term_window`、`dynamic_relationships` 和 `core_anchors`，避免上下文失焦。
- 把角色输出重构为“认知层 + 表现层”的 `AgentAction` 结构，逼模型显露言不由衷。
- 在 `Writer` 之前新增 `Symbolism / Subtext` 中间层，先做意象映射，再写正文。
- 新增 `miro-scifi-novel` 长篇 runner，可串联多个场景直接输出一篇万字级小说。

## 当前架构

```text
[Start]
  |
  v
[Showrunner Node] -> 生成节拍表 / 目标结局 / 强制事件
  |
  v
[Director Setup]
  |
  v
[Character A]
  |
  v
[Character B]
  |
  v
[Director Checkpoint] -> 资源衰减 / 关系更新 / 强制拉回主线
  |
  +---- 回合未结束 ----> [Character A]
  |
  +---- 回合结束 ------> [Symbolism Node] -> [Writer Node] -> [End]
```

## 安装

### 方式 1：使用 `uv`（推荐）

```bash
uv run python -m miro_scifi --mode mock
```

### 方式 2：使用传统虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

如果你的系统缺少 `ensurepip` / `python3-venv`，优先使用 `uv`。

## 运行

### 1) 本地演示

```bash
uv run python -m miro_scifi --mode mock
```

### 2) 真实模型

```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
uv run python -m miro_scifi --mode live \
  --character-model openai:gpt-4o \
  --showrunner-model openai:gpt-4o \
  --symbolism-model openai:gpt-4o \
  --writer-model anthropic:claude-3-5-sonnet-latest
```

## 新的核心状态

- `resource_state`：角色资源池，按回合自动衰减。
- `showrunner_plan`：场景节拍表，包含目标结局、冲突、强制事件、伏笔。
- `short_term_window`：只保留最近几轮公开窗口。
- `dynamic_relationships`：角色对彼此的动态印象标签。
- `core_anchors`：无论对话多长都始终注入 Prompt 顶部的创伤 / 渴望 / 伪装。
- `symbolism_plan`：潜台词与物象建议。
- `chapter_text`：最终小说正文。

## 关键文件

- `src/miro_scifi/models.py`：新的 Pydantic 数据模型与全局状态。
- `src/miro_scifi/prompts.py`：Showrunner、Character、Symbolism、Writer 全套 Prompt。
- `src/miro_scifi/engine.py`：Character / Showrunner / Symbolism 的 live + mock 引擎。
- `src/miro_scifi/graph.py`：资源衰减、关系压缩、状态机编排。
- `src/miro_scifi/writer.py`：场景打包、潜台词指导、文学渲染。
- `src/miro_scifi/main.py`：CLI 入口。

## 长篇运行

```bash
uv run python -m miro_scifi.novel_runner --mode live
```

默认会读取 `~/MyInvestment/.env.local`，并把 `SILICONFLOW_API_KEY` 自动映射到 OpenAI 兼容环境变量，输出：

- `outputs/echo_tax_idea.md`
- `outputs/echo_tax_novel.md`
- `outputs/echo_tax_states/`

```bash
uv run python -m miro_scifi.one_shot_novel --max-chapters 1
```

如果你只是想尽快产出一版长篇草稿，优先用 `one_shot_novel`；如果你要验证多节点架构，再用 `novel_runner`。

## `novel_runner` 新 idea

- `echo_tax`：原始的《回声税》长篇方案。
- `cry_guarantee`：新的《哭声担保》方案，聚焦“公开哀悼权”的抵押与静音处理。

示例：

```bash
uv run python -m miro_scifi.novel_runner --mode live --idea cry_guarantee --max-scenes 2
```

续跑示例：

```bash
uv run python -m miro_scifi.novel_runner --mode live --idea cry_guarantee --start-scene 3 --max-scenes 1
```

如果 `states_dir` 里已经有前一场的 `scene_XX.json`，runner 会自动续接资源状态、关系标签和锚点。
