# HelloAgents智能旅行助手 🌍✈️

基于HelloAgents框架构建的智能旅行规划助手,集成高德地图MCP服务,提供个性化的旅行计划生成。

## ✨ 功能特点

- 🤖 **AI驱动的旅行规划**: 基于HelloAgents框架的SimpleAgent,智能生成详细的多日旅程
- 🗺️ **高德地图集成**: 通过MCP协议接入高德地图服务,支持景点搜索、路线规划、天气查询
- 🧠 **智能工具调用**: Agent自动调用高德地图MCP工具,获取实时POI、路线和天气信息
- 🎨 **现代化前端**: Vue3 + TypeScript + Vite,响应式设计,流畅的用户体验
- 📱 **完整功能**: 包含住宿、交通、餐饮和景点游览时间推荐

## 🏗️ 技术栈

### 后端
- **框架**: HelloAgents (基于SimpleAgent)
- **API**: FastAPI
- **MCP工具**: amap-mcp-server (高德地图)
- **LLM**: 支持多种LLM提供商(OpenAI, DeepSeek等)

### 前端
- **框架**: Vue 3 + TypeScript
- **构建工具**: Vite
- **UI组件库**: Ant Design Vue
- **地图服务**: 高德地图 JavaScript API
- **HTTP客户端**: Axios

## 📁 项目结构

```
helloagents-trip-planner/
├── backend/                    # 后端服务
│   ├── app/
│   │   ├── agents/            # Agent实现
│   │   │   └── trip_planner_agent.py
│   │   ├── api/               # FastAPI路由
│   │   │   ├── main.py
│   │   │   └── routes/
│   │   │       ├── trip.py
│   │   │       └── map.py
│   │   ├── services/          # 服务层
│   │   │   ├── amap_service.py
│   │   │   └── llm_service.py
│   │   ├── models/            # 数据模型
│   │   │   └── schemas.py
│   │   └── config.py          # 配置管理
│   ├── requirements.txt
│   ├── .env.example
│   └── .gitignore
├── frontend/                   # 前端应用
│   ├── src/
│   │   ├── components/        # Vue组件
│   │   ├── services/          # API服务
│   │   ├── types/             # TypeScript类型
│   │   └── views/             # 页面视图
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 🚀 快速开始

### 前提条件

- Python 3.10+
- Node.js 16+
- 高德地图API密钥 (Web服务API和Web端(JS API))
- LLM API密钥 (OpenAI/DeepSeek等)

### 后端安装

1. 进入后端目录
```bash
cd backend
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件,填入你的API密钥
```

5. 启动后端服务
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 前端安装

1. 进入前端目录
```bash
cd frontend
```

2. 安装依赖
```bash
npm install
```

3. 配置环境变量
```bash
# 创建.env文件, 填入高德地图Web API Key 和 Web端JS API Key
cp .env.example .env
```

4. 启动开发服务器
```bash
npm run dev
```

5. 打开浏览器访问 `http://localhost:5173`

## 📝 使用指南

1. 在首页填写旅行信息:
   - 目的地城市
   - 旅行日期和天数
   - 交通方式偏好
   - 住宿偏好
   - 旅行风格标签

2. 点击"生成旅行计划"按钮

3. 系统将:
   - 调用HelloAgents Agent生成初步计划
   - Agent自动调用高德地图MCP工具搜索景点
   - Agent获取天气信息和路线规划
   - 整合所有信息生成完整行程

4. 查看结果:
   - 每日详细行程
   - 景点信息与地图标记
   - 交通路线规划
   - 天气预报
   - 餐饮推荐

## 🔧 核心实现

### HelloAgents Agent集成

```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool

# 创建高德地图MCP工具
amap_tool = MCPTool(
    name="amap",
    server_command=["uvx", "amap-mcp-server"],
    env={"AMAP_MAPS_API_KEY": "your_api_key"},
    auto_expand=True
)

# 创建旅行规划Agent
agent = SimpleAgent(
    name="旅行规划助手",
    llm=HelloAgentsLLM(),
    system_prompt="你是一个专业的旅行规划助手..."
)

# 添加工具
agent.add_tool(amap_tool)
```

### MCP工具调用

Agent可以自动调用以下高德地图MCP工具:
- `maps_text_search`: 搜索景点POI
- `maps_weather`: 查询天气
- `maps_direction_walking_by_address`: 步行路线规划
- `maps_direction_driving_by_address`: 驾车路线规划
- `maps_direction_transit_integrated_by_address`: 公共交通路线规划

## 📄 API文档

启动后端服务后,访问 `http://localhost:8000/docs` 查看完整的API文档。

主要端点:
- `POST /api/trip/plan` - 生成旅行计划
- `GET /api/map/poi` - 搜索POI
- `GET /api/map/weather` - 查询天气
- `POST /api/map/route` - 规划路线

## 🤝 贡献指南

欢迎提交Pull Request或Issue!

## 📜 开源协议

CC BY-NC-SA 4.0

## 🙏 致谢

- [HelloAgents](https://github.com/datawhalechina/Hello-Agents) - 智能体教程
- [HelloAgents框架](https://github.com/jjyaoao/HelloAgents) - 智能体框架
- [高德地图开放平台](https://lbs.amap.com/) - 地图服务
- [amap-mcp-server](https://github.com/sugarforever/amap-mcp-server) - 高德地图MCP服务器

---

**HelloAgents智能旅行助手** - 让旅行计划变得简单而智能 🌈



## 🧭 阶段A改造说明（LangChain + LangGraph）

阶段A已完成“执行框架迁移”：后端新增 `langgraph` 运行时，并保留 `helloagents` 兼容运行时。

### 新增能力

- 默认使用 `langgraph` 执行旅行规划流程（景点 -> 天气 -> 酒店 -> 规划）。
- 前端可选择运行时：`langgraph`（默认）或 `helloagents`（兼容）。
- API入参新增 `runtime` 字段，前后端已对齐。

### 关键文件

- `backend/app/agents/langgraph_trip_planner.py`：LangGraph 主流程与状态定义。
- `backend/app/services/langchain_service.py`：LangChain LLM 初始化。
- `backend/app/agents/runtime_manager.py`：运行时路由与单例管理。
- `backend/app/api/routes/trip.py`：根据 `runtime` 选择后端运行时。
- `backend/app/models/schemas.py`：`TripRequest` 新增 `runtime` 字段。
- `frontend/src/views/Home.vue`：新增“规划引擎”选择器。
- `frontend/src/types/index.ts`：`TripFormData` 新增 `runtime` 类型。

### Lang 系列知识点映射

- **LangChain**：`ChatOpenAI` 统一模型调用封装。
- **LangGraph**：使用 `StateGraph` 描述节点状态流转和可扩展工作流。
- **可迁移架构**：通过 `runtime_manager` 并存新旧实现，支持灰度切换。

## 🛠️ 本地安装与运行（阶段A）

### 1. 你需要准备的内容

- 高德地图 API Key（`AMAP_API_KEY`）
- LLM Key（`LLM_API_KEY` 或 `OPENAI_API_KEY`）
- 可选：自定义模型地址（`LLM_BASE_URL`）与模型名（`LLM_MODEL_ID`）

### 2. 后端

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 在 .env 中填写 AMAP_API_KEY 与 LLM_API_KEY
python run.py
```

### 3. 前端

```bash
cd frontend
npm install
# 可选: 配置 VITE_API_BASE_URL，默认 http://localhost:8000
npm run dev
```

### 4. 使用方式

- 打开前端后，在“规划引擎”下拉框选择：
  - `LangGraph (阶段A默认)`
  - `HelloAgents (兼容模式)`
- 提交后会调用 `POST /api/trip/plan`，并携带 `runtime` 字段。

## 🚢 部署建议（阶段A）

- 建议把后端容器与前端容器分离部署。
- 后端需注入环境变量：`AMAP_API_KEY`、`LLM_API_KEY`（必填）。
- 若使用企业模型网关，设置 `LLM_BASE_URL` 与 `LLM_MODEL_ID`。
- 生产环境建议关闭 `reload`，并通过进程管理器（gunicorn/uvicorn workers）运行。

## 🧩 阶段B改造说明（ReAct + MCP 工具编排 + 可观测日志）

阶段B在 LangGraph 的信息获取阶段引入 ReAct（Thought/Action/Observation）模式，并统一使用 `ToolRegistry` 管理地图能力。

### 新增能力

- 信息获取节点改为 ReAct 循环，按步骤执行 `map.search_poi`、`map.weather`、`map.route`。
- 不再依赖字符串硬编码工具调用，改用 LangChain Tool Schema + Pydantic 参数校验。
- 工具层统一支持超时、重试、fallback。
- 增加工具级错误分类：`rate_limit(429)`、`timeout`、`bad_request`、`server_error`。
- 记录每一步 action/observation 轨迹并输出日志，便于调试与经验复用。
- 路线工具会根据前端 `transportation` 自动映射调用：`公共交通->transit`、`自驾->driving`、`步行->walking`、`骑行->riding`；`混合`会额外补充步行路线调用。

### 验收覆盖

- 在 `backend/tests/test_tool_registry.py` 中覆盖 3 类错误降级场景：限流、超时、参数错误。
