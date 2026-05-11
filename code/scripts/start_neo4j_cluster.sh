#!/bin/bash
# Start multiple Neo4j instances for parallel Graphiti execution.
# Usage: ./start_neo4j_cluster.sh [num_instances] [start|stop|status]
#
# Each instance gets bolt port 7687+i, http port 7474+i.
# Set NEO4J_BASE_PORT=7687 when running run_agent.py.

NUM=${1:-10}
ACTION=${2:-start}

if [ "$ACTION" = "stop" ]; then
    for i in $(seq 0 $((NUM-1))); do
        docker stop neo4j-meme-$i 2>/dev/null
        docker rm neo4j-meme-$i 2>/dev/null
    done
    # Also stop the default single instance
    docker stop neo4j-meme 2>/dev/null
    echo "Stopped $NUM Neo4j instances"
    exit 0
fi

if [ "$ACTION" = "status" ]; then
    docker ps --format "{{.Names}}: {{.Status}}" | grep neo4j
    echo "---"
    docker stats --no-stream --format "{{.Name}}: {{.MemUsage}}" | grep neo4j
    exit 0
fi

# Stop default single instance
docker stop neo4j-meme 2>/dev/null
docker rm neo4j-meme 2>/dev/null

echo "Starting $NUM Neo4j instances (128MB heap each)..."
for i in $(seq 0 $((NUM-1))); do
    BOLT_PORT=$((7687 + i))
    HTTP_PORT=$((7474 + i))
    NAME="neo4j-meme-$i"

    docker stop $NAME 2>/dev/null
    docker rm $NAME 2>/dev/null

    docker run -d --name $NAME \
        -p $HTTP_PORT:7474 -p $BOLT_PORT:7687 \
        -e NEO4J_AUTH=neo4j/mempass123 \
        -e 'NEO4J_PLUGINS=["apoc"]' \
        -e NEO4J_server_memory_heap_initial__size=128m \
        -e NEO4J_server_memory_heap_max__size=128m \
        -e NEO4J_server_memory_pagecache_size=32m \
        neo4j:5 > /dev/null 2>&1

    echo "  $NAME: bolt=$BOLT_PORT http=$HTTP_PORT"
done

echo "Waiting 15s for Neo4j instances to start..."
sleep 15

echo "Status:"
docker ps --format "  {{.Names}}: {{.Status}}" | grep neo4j
echo ""
echo "Usage: NEO4J_BASE_PORT=7687 python3 run_agent.py --agent-type graphiti -w $NUM ..."
