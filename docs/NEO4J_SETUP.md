# Neo4j Live Setup for Rhea FinGraph

This guide gets **live Neo4j + Cypher queries** running on your Mac and wired
into the dashboard. The dashboard already renders the *local* temporal graph
(written as PyTorch snapshots, no server needed). This doc covers the **live**
layer: a real Neo4j graph database you query with Cypher from the dashboard.

The dashboard's **Graph panel** has two working views:

1. **Local temporal graph** — always available; reads `artifacts/graph/*`
   snapshot files. No setup needed.
2. **Live Neo4j / Cypher** — appears in the same panel; needs Neo4j running.
   The backend proxy is `POST /api/v1/graph/cypher` and the panel flips to
   `NEO4J LIVE` once the `bolt://localhost:7687` handshake succeeds.

---

## 1. Install Java (required by Neo4j 5.x)

Your Mac has no Java yet. Install OpenJDK 21 via Homebrew:

```bash
brew install openjdk@21
```

Add it to PATH (Homebrew's openjdk is not linked by default):

```bash
sudo ln -sfn /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk \
  /Library/Java/JavaVirtualMachines/openjdk-21.jdk
export JAVA_HOME="/opt/homebrew/opt/openjdk@21"
echo 'export JAVA_HOME="/opt/homebrew/opt/openjdk@21"' >> ~/.zshrc
```

Verify: `java --version` → `openjdk 21.x`.

> If Homebrew is at `/usr/local` (Intel Mac) instead of `/opt/homebrew`
> (Apple Silicon), adjust the paths accordingly.

## 2. Install Neo4j Community

```bash
brew install neo4j
```

This installs the `neo4j` command. Its default database lives under
`$(brew --prefix)/var/neo4j`.

## 3. Start Neo4j

```bash
neo4j start
```

Wait ~10s, then check:

```bash
neo4j status     # -> "Neo4j is running"
```

## 4. (Recommended) Set a known password

The backend's default credentials are in `src/fingraph_sentinel/config.py`:

- `neo4j_url = "bolt://localhost:7687"`
- `neo4j_username = "neo4j"`
- `neo4j_password = "change-me-local-only"`

Fresh Neo4j ships with `neo4j` / `neo4j`. Align it to the backend default:

```bash
cypher-shell -u neo4j -p neo4j "ALTER CURRENT USER SET PASSWORD \
  FROM 'neo4j' TO 'change-me-local-only';"
```

If you keep a different password, update `config.py` (or set an env var) to
match. Tell the agent which you chose.

## 5. Confirm the socket is up

```bash
nc -z localhost 7687 && echo "bolt up on 7687"
curl -s http://localhost:7474   # Neo4j Browser UI (log in with neo4j/pass)
```

Neo4j Browser at http://localhost:7474 is handy for ad-hoc Cypher—but the
dashboard panel runs the queries for you.

## 6. Load the fraud graph

From the repo root:

```bash
make ingest-graph   # src/fingraph_sentinel/graph_ingest.py
```

This streams the splits into Neo4j as `:Customer`, `:Merchant`, `:Card` nodes
with `PURCHASED` / `HAS_CARD` / `SWIPED_AT` edges (~24.39M edges; allow
10–20 min). It is memory-efficient (chunked parquet) and transaction-safe.

## 7. Start the backend + dashboard

```bash
make api-server          # uvicorn on 127.0.0.1:8000
# in a second terminal, from apps/dashboard:
PATH="/usr/local/bin:$PATH" node_modules/.bin/next dev -p 3001
```

Open http://localhost:3001 → **Graph store · Layer 2** panel:

- The status pill flips to **NEO4J ONLINE** within ~15s.
- The **live Cypher console** lists whitelisted queries (connected web,
  high fraud-rate merchants, customers & cards, confirmed-fraud edges).
  Pick one, run it, and it renders as a live force-directed graph.

## Backend Cypher gateway

`POST /api/v1/graph/cypher` with a JSON body `{"query": "<key>", "limit": 100}`.
Only a **whitelisted set** of static queries is allowed — arbitrary Cypher is
rejected (422) for safety; the surface is read-only. When Neo4j is down it
returns an honest offline payload (not a crash).

Whitelisted query keys: `overview`, `hot_merchants`, `cards_of_customers`,
`fraud_edges`.

## Alternative: Docker Desktop

If you prefer containers (needs Docker Desktop installed & running):

```bash
docker run --name rhea-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/change-me-local-only neo4j:5-community
```

## Honesty note

The dashboard never fabricates graph data. If Neo4j is off, the panel shows an
honest `NEO4J OFFLINE` card with the exact command to bring it up. The local
temporal snapshot view works with or without Neo4j.
