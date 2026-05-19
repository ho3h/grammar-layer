#!/usr/bin/env bash
# Bootstrap a project-local Neo4j 2026.03.1 with GDS 2026.04 + APOC 2026.03.
# Adapted from PRD §6.1 step 1: Neo4j Desktop 2 path was infeasible (no API for headless DBMS
# creation), so we instead clone the Homebrew install into .neograph-db/ and run it
# on port 7693, isolated from the user's existing Neo4j instances.
#
# Idempotent — safe to re-run.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB_DIR="$PROJECT_ROOT/.neograph-db"
HOMEBREW_NEO4J="/usr/local/Cellar/neo4j/2026.03.1/libexec"
GDS_VERSION="2026.04.0"
GDS_URL="https://github.com/neo4j/graph-data-science/releases/download/${GDS_VERSION}/neo4j-graph-data-science-${GDS_VERSION}.jar"

export JAVA_HOME=${JAVA_HOME:-/usr/local/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home}

if [ -d "$DB_DIR" ] && [ -f "$DB_DIR/bin/neo4j" ]; then
  echo "[bootstrap] $DB_DIR exists; skipping copy."
else
  echo "[bootstrap] Copying Neo4j from $HOMEBREW_NEO4J to $DB_DIR ..."
  mkdir -p "$DB_DIR"/{conf,data,logs,plugins,import,run,licenses,certificates,products,metrics}
  cp -R "$HOMEBREW_NEO4J/bin" "$DB_DIR/bin"
  cp -R "$HOMEBREW_NEO4J/lib" "$DB_DIR/lib"
  cp -R "$HOMEBREW_NEO4J/labs" "$DB_DIR/labs"
  cp -R "$HOMEBREW_NEO4J/products" "$DB_DIR/products" 2>/dev/null || true
  cp "$HOMEBREW_NEO4J"/{LICENSE.txt,README.txt,UPGRADE.txt,packaging_info,ThirdPartyLicenses.txt} "$DB_DIR/" 2>/dev/null || true
fi

if [ ! -f "$DB_DIR/plugins/apoc-2026.03.1-core.jar" ]; then
  echo "[bootstrap] Installing APOC core ..."
  cp "$HOMEBREW_NEO4J/labs/apoc-2026.03.1-core.jar" "$DB_DIR/plugins/"
fi

if [ ! -f "$DB_DIR/plugins/neo4j-graph-data-science-${GDS_VERSION}.jar" ]; then
  echo "[bootstrap] Downloading GDS ${GDS_VERSION} ..."
  curl -sSL -o "$DB_DIR/plugins/neo4j-graph-data-science-${GDS_VERSION}.jar" "$GDS_URL"
fi

if [ ! -f "$DB_DIR/conf/neo4j.conf.applied" ]; then
  echo "[bootstrap] Writing neograph neo4j.conf ..."
  cat > "$DB_DIR/conf/neo4j.conf" <<'EOF'
server.default_listen_address=127.0.0.1
server.bolt.enabled=true
server.bolt.listen_address=:7693
server.bolt.advertised_address=:7693
server.http.enabled=true
server.http.listen_address=:7479
server.http.advertised_address=:7479
server.https.enabled=false

server.memory.heap.initial_size=4G
server.memory.heap.max_size=16G
server.memory.pagecache.size=4G

dbms.security.auth_enabled=true
dbms.security.procedures.unrestricted=apoc.*,gds.*
dbms.security.procedures.allowlist=apoc.*,gds.*

server.jvm.additional=-Dlog4j2.formatMsgNoLookups=true
server.jvm.additional=-XX:-OmitStackTraceInFastThrow
server.jvm.additional=-XX:+AlwaysPreTouch
server.jvm.additional=-XX:+UnlockExperimentalVMOptions
server.jvm.additional=-XX:+TrustFinalNonStaticFields
server.jvm.additional=-XX:+DisableExplicitGC
server.jvm.additional=-XX:-RestrictContended
server.jvm.additional=-Djava.awt.headless=true
server.jvm.additional=-Dunsupported.dbms.udc.source=neograph

dbms.usage_report.enabled=false
dbms.tx_log.rotation.retention_policy=1 days
EOF
  touch "$DB_DIR/conf/neo4j.conf.applied"
fi

if [ ! -f "$DB_DIR/data/dbms/auth" ]; then
  echo "[bootstrap] Setting initial password ..."
  "$DB_DIR/bin/neo4j-admin" dbms set-initial-password neograph_local_dev
fi

if lsof -iTCP:7693 -sTCP:LISTEN -P 2>/dev/null | grep -q LISTEN; then
  echo "[bootstrap] Neo4j already listening on 7693."
else
  echo "[bootstrap] Starting Neo4j ..."
  nohup "$DB_DIR/bin/neo4j" console > "$DB_DIR/logs/console.log" 2>&1 &
  until lsof -iTCP:7693 -sTCP:LISTEN -P 2>/dev/null | grep -q LISTEN; do
    sleep 2
  done
  echo "[bootstrap] Neo4j listening on 7693."
fi

echo "[bootstrap] Done. Verify with:"
echo "  cypher-shell -a bolt://localhost:7693 -u neo4j -p neograph_local_dev \"RETURN gds.version();\""
