#!/usr/bin/env bash
# mem_monitor.sh

LOG_DIR="/var/log/memory_history"
CHECK_INTERVAL=2         # 检查间隔（秒）
MAX_LOG_SIZE_BYTES=$((10 * 1024 * 1024))  # 单个日志最大 10MB
ALERT_THRESHOLD=85        # 触发告警的内存使用百分比（整数）
LOG_FILE="$LOG_DIR/memory_$(date +%Y%m%d).log"

mkdir -p "$LOG_DIR"

rotate_log_if_needed() {
  if [ -f "$LOG_FILE" ]; then
    local size
    size=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
    if [ "$size" -gt "$MAX_LOG_SIZE_BYTES" ]; then
      local ts
      ts=$(date +%H%M%S)
      mv "$LOG_FILE" "${LOG_FILE}.${ts}"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] 日志轮转: ${LOG_FILE} -> ${LOG_FILE}.${ts}" >> "${LOG_FILE}.${ts}"
    fi
  fi
}

log_memory_info() {
  local ts mem_percent_int mem_percent_str
  ts=$(date '+%Y-%m-%d %H:%M:%S')

  # 计算内存使用率（百分比，取整）
  # free 输出第2行：总内存/已用内存
  mem_percent_int=$(free | awk 'NR==2{printf "%.0f", ($3*100/$2)}')
  mem_percent_str=$(free | awk 'NR==2{printf "%.1f", ($3*100/$2)}')

  {
    echo "[$ts] 内存使用率: ${mem_percent_str}%"
    free -m
    if [ "$mem_percent_int" -gt "$ALERT_THRESHOLD" ]; then
      echo "[$ts] 高内存使用警告: ${mem_percent_str}% >= ${ALERT_THRESHOLD}%"
      echo "=== 前10个内存占用进程 ==="
      ps aux --sort=-%mem | head -11
      echo "=== 系统负载 ==="
      uptime
      echo "================================="
    fi
  } >> "$LOG_FILE"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动内存监控: 日志目录=$LOG_DIR, 间隔=${CHECK_INTERVAL}s, 阈值=${ALERT_THRESHOLD}%" >> "$LOG_FILE"

while true; do
  rotate_log_if_needed
  log_memory_info
  sleep "$CHECK_INTERVAL"
done