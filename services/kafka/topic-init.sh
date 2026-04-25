#!/bin/bash
# ============================================================
# PulseChain Kafka Topic Initializer
# Creates all required topics with proper partition counts.
# Runs once on startup via kafka-init service.
# ============================================================

set -e

BROKER="kafka:9092"
PARTITIONS=3
REPLICATION=1

echo "Waiting for Kafka broker at $BROKER..."
cub kafka-ready -b $BROKER 1 60

echo "Creating PulseChain topics..."

topics=(
  "pulsechain-wastewater"
  "pulsechain-pharmacy"
  "pulsechain-search-trends"
  "pulsechain-absenteeism"
  "pulsechain-ed-triage"
  "pulsechain-dlq"
  "pulsechain-alerts"
  "pulsechain-audit"
)

for topic in "${topics[@]}"; do
  echo "  Creating topic: $topic"
  kafka-topics --bootstrap-server $BROKER \
    --create \
    --if-not-exists \
    --topic "$topic" \
    --partitions $PARTITIONS \
    --replication-factor $REPLICATION \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete
done

echo ""
echo "=== All topics created successfully ==="
kafka-topics --bootstrap-server $BROKER --list
echo "======================================="