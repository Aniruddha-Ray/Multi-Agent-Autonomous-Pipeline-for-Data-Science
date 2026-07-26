# Adaptive Multi-Agent Planning for Autonomous Data Science Pipelines

## Stage 1 — Architecture Specification (No Implementation)

> This document is the \*\*running spec of record\*\*. Every later stage must stay consistent with what's defined here. Naming, contracts, and schemas introduced now are considered frozen unless a later stage explicitly calls out and justifies a change.

\---

## 1\. High-Level Architecture

Clean architecture, strict downward dependency flow. Upper layers depend on abstractions (interfaces/ports) defined by lower layers — never on concrete implementations directly (Dependency Inversion).

```
┌─────────────────────────────────────────────────────────────┐
│  Presentation Layer        (Streamlit UI, WebSocket clients) │
├─────────────────────────────────────────────────────────────┤
│  Application Layer         (FastAPI routers, use-cases,      │
│                              orchestration entrypoints)       │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer               (LangGraph graph, nodes, agents,  │
│                              planner, critic, tools)          │
├─────────────────────────────────────────────────────────────┤
│  Service Layer              (domain services: dataset,        │
│                              ml, memory, report, explain)     │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                 (repositories, ORM models,         │
│                              vector/graph adapters)            │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Layer        (Postgres, Redis, Qdrant, Neo4j,  │
│                              MLflow, Celery, LLM providers,    │
│                              Docker, config, logging)          │
└─────────────────────────────────────────────────────────────┘
```

**Rules enforced across all stages:**

* Agent Layer never talks to the Data Layer directly — only through Service Layer interfaces.
* Service Layer depends on **repository interfaces** (`ports/`), not concrete SQLAlchemy/Qdrant/Neo4j classes. Concrete adapters live in Data/Infrastructure and are injected.
* All cross-layer contracts are Pydantic models defined in `backend/app/contracts/` (application-facing) and `agents/schemas/` (agent-facing). No layer invents its own ad-hoc dict shapes.
* Dependency Injection container: a single composition root (`backend/app/container.py`) wires concrete implementations to interfaces at startup. Nothing else instantiates infra clients directly.

\---

## 2\. Directory Structure

```
adaptive-ds-agents/
├── backend/
│   └── app/
│       ├── main.py                     # FastAPI app factory
│       ├── container.py                # DI composition root
│       ├── api/
│       │   ├── v1/
│       │   │   ├── datasets.py
│       │   │   ├── pipelines.py
│       │   │   ├── agents.py
│       │   │   ├── memory.py
│       │   │   ├── reports.py
│       │   │   ├── experiments.py
│       │   │   ├── auth.py
│       │   │   └── health.py
│       │   └── deps.py                 # FastAPI Depends() providers
│       ├── contracts/                  # Pydantic request/response DTOs
│       ├── middleware/                 # auth, rate limit, logging, error handlers
│       └── websockets/                 # execution stream, log stream
│
├── agents/
│   ├── graph/
│   │   ├── state.py                    # PipelineState (TypedDict/Pydantic)
│   │   ├── build\_graph.py              # LangGraph graph assembly
│   │   ├── routing.py                  # conditional edge functions
│   │   └── checkpointer.py             # Postgres/Redis checkpoint backend
│   ├── nodes/
│   │   ├── planner\_agent.py
│   │   ├── metadata\_agent.py
│   │   ├── validation\_agent.py
│   │   ├── cleaning\_agent.py
│   │   ├── eda\_agent.py
│   │   ├── feature\_engineering\_agent.py
│   │   ├── model\_recommendation\_agent.py
│   │   ├── training\_agent.py
│   │   ├── hpo\_agent.py
│   │   ├── evaluation\_agent.py
│   │   ├── explainability\_agent.py
│   │   ├── report\_agent.py
│   │   └── critic\_agent.py
│   ├── schemas/                        # per-agent typed I/O (Pydantic)
│   ├── prompts/                        # prompt templates per agent
│   ├── tools/                          # MCP-exposed / LangChain tools
│   └── base/
│       ├── base\_agent.py               # abstract agent contract
│       └── retry\_policy.py
│
├── memory/                             # Cognitive Memory Architecture (see §4)
│   ├── working\_memory/
│   │   ├── pipeline\_state\_view.py      # read/write view over active PipelineState
│   │   ├── active\_context.py           # current agent outputs, scratch context
│   │   └── reasoning\_buffer.py         # in-flight reasoning trace (pre-persist)
│   ├── long\_term\_memory/
│   │   ├── experience\_replay/
│   │   ├── dataset\_metadata\_memory/
│   │   ├── pipeline\_memory/
│   │   ├── failure\_memory/
│   │   ├── semantic\_memory/
│   │   └── reasoning\_memory/
│   ├── retrieval\_engine/
│   │   ├── similarity\_search.py        # Qdrant kNN over embeddings
│   │   ├── confidence\_ranking.py       # recency + success-rate + similarity scoring
│   │   ├── memory\_fusion.py            # merges multi-source recalls, dedupes/conflicts
│   │   └── context\_builder.py          # assembles final MemoryContext for an agent call
│   ├── knowledge\_graph/
│   │   ├── neo4j\_client.py
│   │   ├── schema.py                   # node/edge type definitions
│   │   └── sync.py                     # writes execution history into the graph
│   └── ports/                          # abstract interfaces for all of the above
│
├── services/
│   ├── dataset\_service.py
│   ├── ml\_service.py
│   ├── memory\_service.py               # facade over memory/\* for the Service Layer
│   ├── report\_service.py
│   ├── explainability\_service.py
│   ├── evaluation\_benchmark\_service.py
│   └── ports/                          # interfaces the Agent Layer depends on
│
├── database/
│   ├── models/                         # SQLAlchemy ORM models
│   ├── repositories/                   # concrete repo implementations
│   ├── migrations/                     # Alembic
│   └── session.py
│
├── ml/
│   ├── pipelines/
│   ├── training/
│   ├── optimization/                   # Optuna
│   ├── registry/                       # MLflow model registry wrapper
│   └── feature\_engineering/
│
├── evaluation/
│   ├── explainability/                 # SHAP, LIME, counterfactuals
│   ├── metrics/
│   └── benchmarks/                     # AutoGluon/PyCaret/TPOT comparisons
│
├── report\_generation/
│   ├── templates/
│   ├── pdf\_builder.py
│   ├── html\_builder.py
│   └── markdown\_builder.py
│
├── frontend/
│   └── streamlit\_app/
│       ├── Home.py
│       └── pages/
│           ├── 1\_Upload.py
│           ├── 2\_Dataset\_Explorer.py
│           ├── 3\_Execution\_Monitor.py
│           ├── 4\_Pipeline\_Graph.py
│           ├── 5\_Memory\_Explorer.py
│           ├── 6\_Knowledge\_Graph.py
│           ├── 7\_Experiments.py
│           └── 8\_Reports.py
│
├── config/
│   ├── settings.py                     # Pydantic Settings, env-driven
│   ├── logging\_config.py               # Loguru setup
│   └── llm\_providers.py                # OpenAI/Anthropic/Ollama factory
│
├── infrastructure/
│   ├── llm/                            # provider adapters
│   ├── vector\_store/                   # Qdrant client wrapper
│   ├── graph\_store/                    # Neo4j client wrapper
│   ├── cache/                          # Redis client wrapper
│   ├── task\_queue/                     # Celery app + tasks
│   └── tracing/                        # LangSmith / OpenTelemetry setup
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── deployment/
│   ├── docker/
│   │   ├── backend.Dockerfile
│   │   ├── frontend.Dockerfile
│   │   └── worker.Dockerfile
│   ├── docker-compose.yml
│   ├── docker-compose.prod.yml
│   └── github-actions/
│       └── ci.yml
│
├── docs/
├── research/                           # papers, ablations, benchmark write-ups
├── scripts/                            # one-off ops scripts
├── alembic.ini
├── pyproject.toml
├── requirements.txt
└── .env.example
```

\---

## 3\. Dependency Graph (module-level, allowed import directions only)

```
frontend/streamlit\_app  ──▶  backend/app/api (HTTP/WS only, no direct imports)

backend/app/api  ──▶  backend/app/contracts
backend/app/api  ──▶  services/\*  (via container-injected interfaces)
backend/app/api  ──▶  agents/graph (to invoke the compiled LangGraph)

agents/nodes/\*  ──▶  agents/schemas, agents/prompts, agents/tools
agents/nodes/\*  ──▶  services/ports/\*        (NOT services/\*\_service.py directly)
agents/nodes/\*  ──▶  memory/working\_memory   (direct — working memory is call-scoped,
                                                not a service concern)
agents/graph/\*  ──▶  agents/nodes/\*, agents/graph/state

services/\*\_service.py  ──▶  services/ports/\*, database/repositories (interfaces),
                              memory/ports/\* (long-term + retrieval + KG, NEVER
                              memory/working\_memory), ml/\*, evaluation/\*,
                              report\_generation/\*

memory/retrieval\_engine/\*  ──▶  memory/long\_term\_memory/\*, memory/knowledge\_graph/\*
memory/long\_term\_memory/\*  ──▶  memory/ports, infrastructure/vector\_store
memory/knowledge\_graph/\*   ──▶  memory/ports, infrastructure/graph\_store
memory/working\_memory/\*    ──▶  agents/graph/state   (leaf w.r.t. other memory modules)

database/repositories/\*  ──▶  database/models, database/session

ml/\*, evaluation/\*, report\_generation/\*  ──▶  (leaf; no upward imports)

config/\*, infrastructure/\*  ──▶  (leaf; imported by everything, imports nothing internal)
```

**Hard rule:** no module in `agents/`, `services/`, `database/`, `ml/` imports from `backend/app/\*` or `frontend/\*`. This guarantees the agent/service/data core is runnable and testable headless (critical for pytest + CI), and prevents circular imports.

**New rule from this revision:** `working\_memory` is the only memory submodule the Agent Layer may import directly (it's ephemeral, call-scoped, and cheap — no point routing it through a service). Everything durable (`long\_term\_memory`, `retrieval\_engine`, `knowledge\_graph`) is only reachable through `services/memory\_service.py`, so agents never bypass ranking/fusion logic to hit raw storage.

\---

## 4\. Cognitive Memory Architecture

The memory layer is not a bag of databases — it's structured as a **cognitive memory architecture**, deliberately modeled on the working-memory / long-term-memory / retrieval distinction from human cognition. This is one of the project's core research contributions (Experience Replay + Semantic Memory Retrieval), and the framing matters for the eventual paper, not just the code:

> The framework adopts a cognitive memory architecture inspired by human problem solving. Working Memory maintains the current execution state, while Long-Term Memory stores prior experiences, semantic knowledge, reasoning traces, and successful workflows. A Retrieval Engine selectively recalls relevant experiences using similarity and confidence scoring before constructing the context for subsequent agent decisions.

```
Memory Layer
│
├── Working Memory                      (ephemeral, single-run scope)
│      ├── Pipeline State               → live PipelineState object
│      ├── Current Agent Outputs        → most recent AgentResult per stage
│      ├── Active Reasoning Trace       → in-flight ReasoningStep buffer
│      └── Temporary Context            → scratch values, not persisted past the run
│
├── Long-Term Memory                    (durable, cross-run scope)
│      ├── Experience Replay            → full past run trajectories + outcomes
│      ├── Dataset Metadata             → profiling summaries, past dataset fingerprints
│      ├── Pipeline Memory              → successful end-to-end pipeline configs
│      ├── Failure Memory               → failure modes + what fixed them
│      ├── Semantic Memory              → embedded facts, domain knowledge, distilled rules
│      └── Reasoning Memory             → archived reasoning traces (post-run)
│
├── Retrieval Engine                    (the "cognition" in cognitive memory)
│      ├── Similarity Search            → Qdrant kNN over embeddings
│      ├── Confidence Ranking           → similarity × recency × historical success rate
│      ├── Memory Fusion                → merges/deduplicates multi-source recalls,
│      │                                   resolves conflicting past experiences
│      └── Context Builder              → assembles the final MemoryContext object
│                                          handed to an agent before it acts
│
└── Knowledge Graph                     (structural/relational memory)
       ├── Neo4j
       ├── Dataset Relations            → Dataset --USED/GENERATED--> Feature/Transform
       ├── Pipeline Relations           → Pipeline --OUTPERFORMED/SIMILAR\_TO--> Pipeline
       └── Feature Relations            → Feature --RECOMMENDED--> Model
```

### 4.1 Design rules that follow from this structure

* **Working Memory is never persisted directly.** At the end of a run, the Reasoning Buffer and Current Agent Outputs are *distilled* into Long-Term Memory (Reasoning Memory, Experience Replay) by an explicit consolidation step — mirroring memory consolidation rather than a raw dump. This consolidation step is implemented in Stage 7.
* **No agent queries Long-Term Memory directly.** Every recall goes through the Retrieval Engine, so similarity search, confidence ranking, and fusion are applied uniformly — an agent never sees raw unranked hits from Qdrant.
* **MemoryContext (already in `PipelineState`, §5) is the sole output of the Retrieval Engine.** Its shape is expanded to make the four-part structure explicit:

```python
class MemoryContext(BaseModel):
    working\_memory\_snapshot: WorkingMemorySnapshot
    retrieved\_experiences: List\[RankedMemory]      # from Experience Replay
    retrieved\_semantic\_facts: List\[RankedMemory]   # from Semantic Memory
    retrieved\_pipeline\_templates: List\[RankedMemory]  # from Pipeline Memory
    retrieved\_failure\_warnings: List\[RankedMemory]    # from Failure Memory
    knowledge\_graph\_context: Optional\[KnowledgeGraphSubgraph]
    fusion\_notes: List\[str]                        # conflicts found + how resolved

class RankedMemory(BaseModel):
    memory\_id: UUID
    source: Literal\["experience\_replay","semantic","pipeline","failure","reasoning"]
    content: dict
    similarity\_score: float
    recency\_score: float
    historical\_success\_rate: Optional\[float]
    confidence\_score: float             # fused final rank used for ordering
```

* **Knowledge Graph is written asynchronously**, not on the critical path of an agent's turn — `pipeline\_stage\_executions` rows are the source of truth, and `memory/knowledge\_graph/sync.py` projects them into Neo4j (via a Celery task, wired in Stage 8/12), keeping graph writes decoupled from pipeline latency.
* **`services/memory\_service.py`** is the only Service Layer entry point into `long\_term\_memory`, `retrieval\_engine`, and `knowledge\_graph` — it's a thin facade so the API layer's `/api/v1/memory/\*` endpoints and the Agent Layer both go through one place.

\---

## 5\. Agent Communication Protocol

All agents communicate through one shared, strongly-typed **PipelineState** object passed through the LangGraph graph — never through side-channel globals.

```python
class PipelineState(BaseModel):
    run\_id: UUID
    dataset\_id: UUID
    user\_id: UUID

    problem\_type: Optional\[Literal\["classification","regression","clustering","timeseries","multilabel"]]
    current\_stage: str
    stage\_history: List\[StageExecutionRecord]

    dataset\_metadata: Optional\[DatasetMetadata]
    validation\_report: Optional\[ValidationReport]
    cleaning\_report: Optional\[CleaningReport]
    eda\_report: Optional\[EDAReport]
    feature\_engineering\_report: Optional\[FeatureEngineeringReport]
    model\_recommendations: Optional\[List\[ModelRecommendation]]
    training\_report: Optional\[TrainingReport]
    hpo\_report: Optional\[HPOReport]
    evaluation\_report: Optional\[EvaluationReport]
    explainability\_report: Optional\[ExplainabilityReport]

    critic\_feedback: List\[CriticVerdict]
    retry\_counts: Dict\[str, int]

    memory\_context: MemoryContext          # retrieved via the Retrieval Engine — see §4
    reasoning\_trace: List\[ReasoningStep]   # append-only, one per agent decision
    cost\_ledger: CostLedger                # tokens + $ + wall-clock per stage

    requires\_human\_approval: bool
    human\_approval\_status: Optional\[Literal\["pending","approved","rejected"]]

    errors: List\[AgentError]
    status: Literal\["running","paused","completed","failed"]
```

**Per-agent contract** (every node implements this shape):

```python
class BaseAgent(ABC):
    name: str
    max\_retries: int
    timeout\_seconds: int

    @abstractmethod
    async def run(self, state: PipelineState) -> AgentResult: ...

    @abstractmethod
    def build\_prompt(self, state: PipelineState) -> str: ...

    @abstractmethod
    def parse\_output(self, raw\_output: str) -> BaseModel: ...

class AgentResult(BaseModel):
    agent\_name: str
    output: BaseModel                # agent-specific typed schema
    confidence\_score: float
    reasoning\_trace: ReasoningStep
    cost: StageCost
    success: bool
    error: Optional\[AgentError]
    next\_suggested\_stage: Optional\[str]   # used by Dynamic Workflow Routing
```

**Routing:** the Planner Agent and `agents/graph/routing.py` conditional-edge functions read `next\_suggested\_stage`, `critic\_feedback`, and `retry\_counts` from state to decide the next node — this is the seam where Adaptive Planning, Cost-aware Planning, and Dynamic Workflow Routing plug in. Concrete routing logic is deferred to Stage 5.

**Critic loop:** after every agent, the Critic Agent may emit a `CriticVerdict` with `action: Literal\["accept","retry","reroute","escalate\_to\_human"]`. This is the seam for Self-Correcting Critic behavior — implemented in Stage 6, wired into the graph in Stage 5.

**Human-in-the-loop:** `requires\_human\_approval` + `human\_approval\_status` back a LangGraph `interrupt` node. Only implemented (not just modeled) in Stage 5.

\---

## 6\. API Contracts (surface only — bodies implemented in Stage 3)

|Method|Path|Purpose|
|-|-|-|
|POST|`/api/v1/auth/token`|issue JWT|
|GET|`/api/v1/health`|liveness/readiness (checks Postgres, Redis, Qdrant, Neo4j, MLflow)|
|POST|`/api/v1/datasets`|upload dataset (multipart), returns `dataset\_id`|
|GET|`/api/v1/datasets/{id}`|metadata + profiling summary|
|POST|`/api/v1/pipelines`|start a run for a dataset, returns `run\_id`|
|GET|`/api/v1/pipelines/{run\_id}`|current `PipelineState` snapshot|
|POST|`/api/v1/pipelines/{run\_id}/approve`|resolve a human-in-the-loop interrupt|
|WS|`/api/v1/pipelines/{run\_id}/stream`|live state/log stream|
|GET|`/api/v1/agents/{run\_id}/trace`|full reasoning trace|
|GET|`/api/v1/memory/search`|semantic memory query (routed through Retrieval Engine)|
|GET|`/api/v1/memory/knowledge-graph`|graph query proxy|
|GET|`/api/v1/experiments/{run\_id}`|MLflow experiment detail|
|GET|`/api/v1/reports/{run\_id}`|generated report (pdf/html/md)|

All request/response bodies are Pydantic DTOs in `backend/app/contracts/`, distinct from the internal `PipelineState` sub-schemas (API layer never leaks internal agent schemas directly — DTOs map to/from them).

\---

## 7\. Database Schema (entities only — DDL/Alembic in Stage 2)

Core tables: `users`, `datasets`, `dataset\_versions`, `pipeline\_runs`, `pipeline\_stage\_executions`, `agent\_executions`, `critic\_verdicts`, `reasoning\_traces`, `experiments` (MLflow run linkage), `models`, `model\_versions`, `reports`, `memory\_records` (metadata only — vectors live in Qdrant), `knowledge\_graph\_sync\_log`, `audit\_log`.

Conventions frozen now:

* Every table: `id UUID PK`, `created\_at`, `updated\_at`, soft-delete via `deleted\_at`.
* `pipeline\_runs.id = PipelineState.run\_id` (same UUID, single source of truth).
* All history tables (`dataset\_versions`, `pipeline\_stage\_executions`, `agent\_executions`) are append-only — updates create new rows, never mutate.
* `memory\_records` gains a `memory\_type` column (`experience\_replay | dataset\_metadata | pipeline | failure | semantic | reasoning`) and a `confidence\_score` column so the Retrieval Engine's ranking has a durable, queryable counterpart to the Qdrant similarity score.
* FK + index policy, exact DDL: **Stage 2**.

\---

## 8\. Configuration Management

`config/settings.py` — one `Settings(BaseSettings)` class, nested sub-settings per concern (`DatabaseSettings`, `RedisSettings`, `QdrantSettings`, `Neo4jSettings`, `LLMSettings`, `MLflowSettings`, `CelerySettings`, `SecuritySettings`). Loaded once from `.env`, injected via the DI container — no module reads `os.environ` directly outside this file.

`LLMSettings` supports `provider: Literal\["openai","anthropic","ollama"]` with per-provider model name, base URL, and key — provider selection is a config swap, not a code branch, at call sites (factory pattern in `config/llm\_providers.py`, implemented Stage 2/5).

\---

## 9\. What's Deliberately Deferred

No code yet. Nothing above is implemented — this stage fixes names, shapes, and boundaries so every later stage plugs into the same seams:

* Adaptive Planning / Dynamic Routing → `agents/graph/routing.py`, driven by `AgentResult.next\_suggested\_stage`
* Self-Correcting Critic → `critic\_agent.py` + `CriticVerdict.action`
* Cognitive Memory Architecture (Working / Long-Term / Retrieval Engine / Knowledge Graph) → `memory/\*`, unified by one `MemoryContext` shape — full implementation in **Stage 7**
* Cost-aware Planning → `CostLedger` + `StageCost`, populated by every agent
* Human-in-the-loop → `requires\_human\_approval` + LangGraph interrupt node
* Reasoning Trace Storage → `ReasoningStep` list, persisted to `reasoning\_traces` table, consolidated into Reasoning Memory, embedded into Qdrant
* Pipeline Knowledge Graph → Neo4j sync from `pipeline\_stage\_executions`, decoupled via Celery
* Meta-learning across datasets → Planner queries `dataset\_metadata\_memory` + `pipeline\_memory` through the Retrieval Engine, not directly

\---

## Next Step

**Stage 2**: Infrastructure — `config/settings.py`, Loguru setup, Docker Compose (Postgres, Redis, Qdrant, Neo4j, MLflow), Alembic init + first migration matching the schema above.

Confirm this Stage 1 spec (or flag changes) before I proceed to Stage 2.

