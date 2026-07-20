# Seshat UI Testing Guide

**Scenario:** A small team is building a new service. The company has a Microsoft agreement, so SQL Server is the standing default. But on Wednesday (2026-05-13) the team prefers Postgres — creating a conflict with the existing policy. Next Wednesday (2026-05-20) they resolve the staging risk and decide on Redis for caching. The Terraform provisioning topic gets dropped when someone has to rush to another meeting, and is only captured as a manual node two days later.

<details>
<summary><strong>Pre-requisites</strong> (expand to set up)</summary>

### 1. Copy the env files

```bash
cp docker/.env.docker.example .env.docker
cp .env.example .env
```

**`.env`** serves two purposes:

1. **Docker Compose variable substitution** — Docker Compose loads `.env` automatically before evaluating `docker-compose.yml`. It provides infrastructure variables: Postgres credentials, port bindings (`API_PORT`, `MLFLOW_PORT`, `LOCALSTACK_PORT`), S3 bucket name, AWS region, and the container-internal `DATABASE_URL` used by the `migrate` service.
2. **Local Python runs** — loaded via `python-dotenv` when you run things outside Docker (`uv run pytest`, `seshat eval harness`, scripts). It adds localhost connection overrides on top of the infra vars: Postgres on `localhost` instead of the `postgres` service name, `SECRETS__PROVIDER=env` so secrets are read directly from env vars, and the API keys themselves.

**`.env.docker`** is injected explicitly into the `seshat-api` and `localstack` containers at runtime. It holds API keys (seeded into LocalStack Secrets Manager at startup so `seshat-api` can fetch them) and Seshat config overrides. Docker Compose does **not** load this file automatically — it is referenced via `env_file:` in the service definitions.

> **The API keys appear in both files — keep them in sync.** `.env` needs them for local runs outside Docker; `.env.docker` needs them for the running containers. If you rotate a key, update both.

### 2. Populate `.env.docker`

Fill in the keys your stack actually uses. At minimum:

| Variable | Purpose |
|----------|---------|
| `SESHAT_ROOT_API_KEY` | Root key for the Admin Panel (any non-empty string works for local testing) |
| `OPENAI_API_KEY` | Embeddings and grounding LLM (default config uses OpenAI) |
| `ANTHROPIC_API_KEY` | Identification and resolution agents (default config uses Anthropic) |
| `ASSEMBLYAI_API_KEY` | Audio transcription |

Leave `dummy-key` in any provider you are not using — the service will start but those features will fail.

> **Alternative providers:** If you have AWS access, you can replace `ANTHROPIC_API_KEY` with Bedrock by setting `EXTRACTION__IDENTIFICATION__PROVIDER=bedrock_converse`, `EXTRACTION__RESOLUTION__PROVIDER=bedrock_converse`, and `AWS_PROFILE` (or ambient AWS credentials). Similarly, you can replace `OPENAI_API_KEY` with Azure OpenAI by setting the `azure_openai` provider for the embedding and grounding config vars and uncommenting the `AZURE_OPENAI_*` variables in `.env.docker`.

### 3. Start the stack

```bash
docker compose up
```

Wait until all services are healthy. The UI is at `http://localhost:8501`.

</details>

---

## 0. Get an API key

The UI requires an API key for everything except the Admin Panel. The root key (`SESHAT_ROOT_API_KEY` in `.env.docker`) only opens the Admin Panel; you need a regular key for the main screens.

1. Open the sidebar → enter the root key in the **Root Key** field.
2. Click **Admin panel** → **Create key** tab.
3. Set a `user_id`, select role `admin` (so you can test all features). Click **Create**.
4. Copy the plaintext key — it is shown only once.

Enter it in the **API Key** field in the sidebar. You should see your user ID resolve and the three nav buttons appear.

---

## 1. Create a manual node

Navigate to **Manual Actions → Nodes → Create**.

This node represents a standing company-wide policy established before the project started. Use **2026-04-23** as the meeting date:

| Field | Value |
|-------|-------|
| Type | Decision |
| Title | Use SQL Server as the default relational store |
| Description | All new services must use SQL Server under the company's Microsoft enterprise agreement. Deviations require explicit architecture board approval. |
| Auto-resolve relationships | Off |

Click **Create**. Copy the returned node ID — you'll use it in steps 1b and 6.

### 1b. Add a dependent action item (optional)

This step adds a second node that depends on the SQL Server policy — an action item to select the SQL Server edition and acquire the required licenses. It seeds an inbound `depends_on` edge on the policy node, making the Impact Traversal in step 5 richer.

You need the SQL Server node ID from step 1. Use the same meeting date (**2026-04-23**):

| Field | Value |
|-------|-------|
| Type | Action Item |
| Title | Select SQL Server edition and acquire enterprise licenses |
| Description | Evaluate SQL Server Standard vs. Enterprise editions based on projected load and feature requirements. Coordinate with procurement to acquire licenses under the existing Microsoft agreement before the first service goes to staging. |
| Auto-resolve relationships | **Off** |

Before clicking **Create**, expand **Manual Relationships** and add one entry:

| Type | Target node ID |
|------|---------------|
| `depends_on` | *(SQL Server policy node ID from step 1)* |

Click **Create**. The relationship is written immediately without calling the resolution agent. In step 5, loading the **inbound** impact of the SQL Server node will now show both this action item and the conflicting Postgres decision from the audio job.

---

## 2. Submit the audio job (Monday's meeting)

Navigate to **Jobs → Submit**.

| Field | Value |
|-------|-------|
| Meeting date | 2026-05-13 |
| Source type | audio |
| Confidence threshold | 1.05 $^{(*)}$ |
| Auto mode | Off |
| File | `data/fixtures/audio/new_service_2026_05_13.mp3` $^{(**)} |

Click **Submit**. The progress bar auto-refreshes through: `pending → transcribing → extracting → awaiting_review`.

(*): Scores are in [0, 1]. Setting above 1 forces all nodes into manual review.

(**): This synthetic meeting recording was generated with the `scripts/generate_synthetic_audio.py` script, which uses Eleven Labs TTS models via their SDK. You need to install the `audio` dependency group as well as an `ELEVENLABS_API_KEY` to re-generate it: `uv run --group audio python scripts/generate_synthetic_audio.py`.

### What to expect

The meeting covers: picking Postgres over MySQL (the team's preference), assigning Anika to set up schema migrations by Friday, deferring the caching decision (no load numbers yet), and a risk that staging may still be down on Thursday.

Expect 3–5 nodes: a decision (Postgres), an action item (Anika's migrations), an open question (caching layer), and a risk (staging timeline).

Because a SQL Server policy node already exists in the KB, the resolution agent may flag a **conflict** between the Postgres decision and the SQL Server policy.

### Review the audio job

Once status is `awaiting_review`:

1. Expand the **transcript** section and skim it.
2. Expand each node. Check that quote anchors line up with the transcript text.
3. Set decisions:
   - **Approve** the Postgres decision, the migrations action item, and the staging risk.
   - **Edit** the risk node title to be more precise (e.g. `"Staging unavailable before Friday deadline"`), then approve.
   - **Reject** any node that looks spurious.
4. Click **Submit decisions**. The job moves to `writing` → `done`.

> After the job completes (successfully) you will see a link to the MLflow run that tracks the job. Clicking on it opens
the MLflow UI, which is the observability layer and contains usage and latency metrics, as well as the traces corresponding
to the LLM calls made by the agents.

Click **View KB →** and note any conflict edges shown for the Postgres decision.

---

## 3. Submit the text job (Wednesday's follow-up)

Navigate back to **Jobs**, and click the **Submit another job** button.

| Field | Value |
|-------|-------|
| Meeting date | 2026-05-20 |
| Source type | text |
| Confidence threshold | 0.50 $^{(*)}$ |
| Auto mode | **On** |
| File | `data/fixtures/text/new_service_followup_2026_05_20.yaml` |

With auto mode on, the job skips the manual review step — all nodes above the confidence threshold are approved automatically. The job goes `pending → extracting → writing → done`.

(*): The auto mode with a low confidence threshold is not recommended in general but intentional here. If any spurious nodes and/or relationships are introduced, go to the **Manual Actions** page and delete them manually.

### What to expect

The follow-up resolves items from Monday: staging is back up, migrations validated, Redis chosen as the caching layer (Memcached ruled out due to session state requirements), and Riley takes a connection pool sizing action item.

Expect 3–4 nodes automatically approved: a decision (Redis), an action item (Riley's pool sizing), and possibly a node resolving the staging risk. The resolution agent should also link these back to the Monday meeting nodes.

---

## 4. Add a manual node with auto-resolve

Navigate to **Manual Actions → Nodes → Create**.

Redis provisioning came up at the end of Wednesday's meeting but got dropped when Anika had to leave. Two days later, Liam finally captures it. Use **2026-05-22** as the meeting date:

| Field | Value |
|-------|-------|
| Type | Action Item |
| Title | Provision Redis infrastructure via Terraform module |
| Description | Set up the Redis caching layer using the shared Terraform module for managed Redis. Configure cluster mode, eviction policy, and set retention appropriate for session state. |
| Auto-resolve relationships | **On** |

Click **Create**. The system calls the resolution agent immediately and displays any inferred relationships. Expect it to link this action item to the Redis decision from the text job.

---

## 5. Explore the graph

Navigate to **Graph → Browse**.

- Leave filters at defaults. Click **Browse**. All nodes from both jobs plus both manual nodes should appear.
- Filter by `ingestion_source = manual`. Confirm the SQL Server policy and the Terraform action item are there.
- Filter by `ingestion_source = pipeline`. Check that audio and text job nodes have confidence scores.

### Search

- Search `"caching layer"` — the Redis decision and the deferred open question from the audio job should rank high.
- Search `"migration"` — Anika's action item and the staging risk should appear.

Note that you can change the search mode. Test both and compare results.

### Impact

- In the **Search** tab, look for "sql server", expand the node and click "Load impact".
- Go to the "Impact tab". The "Node ID" will be pre-filled. Change direction from "outbound" to "inbound" and hit "Load".
- You can explore the node "Impact Traversal", i.e., all nodes that directly or transitively point to the selected node through relationship edges — in this case, the Postgres decision that conflicts with the SQL Server policy, and any further nodes that link back through that chain.
- You can also inspect the Impact Traversal plot hover data.
- You can even click on any other node in the Impact traversal graph, and it will re-load with the new plot being centered there. If no nodes appear, you can re-load changing the direction value.

#### What to expect

The decision on using PostgreSQL conflicts with the decision on using SQL Server. This is unidirectional because the SQL Server node is older, and resolution happens only from ingested to existing nodes.

---

## 6. Override a pipeline node

Navigate to **Manual Actions → Nodes → Edit**.

Pick the staging risk node from the audio job (you can browse all nodes to find its ID). Provide:
- A revised description: `"Staging was down going into the Friday deadline. Resolved 2026-05-19 after infra patched it."`
- Reason: `"Updated with resolution outcome from follow-up meeting"`

It will fail, since only manual nodes can be editted. Repeat the process in the **Override** tab.

---

## 7. Re-ingest the audio job (optional)

Re-ingesting replaces all approved pipeline nodes from a prior run with a fresh extraction.

1. Navigate to **Jobs → Submit**.
2. Submit the audio fixture again with the same meeting date (`2026-05-13`).
3. The UI returns a **409** — the file was already ingested.
4. Toggle **Force re-ingest** on (requires `admin` role).
5. Submit again. Prior pipeline nodes for that job are hard-deleted and a new job starts.
