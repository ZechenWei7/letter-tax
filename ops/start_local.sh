#!/usr/bin/env bash
# 本机重启后一条命令恢复只读监控：watcher（每 10 min 从 pod 拉曲线 / 退出码）+ 看板（每 120 s 重生成 HTML）。
# 都只读 pod、不碰实验；pod 上的链与 pod_watchdog 不受本机影响。
#   bash ops/start_local.sh
cd "$(dirname "$0")/.." || exit 1
mkdir -p cloud_pull
for pat in "setup/auto_pull.sh" "ops/dashboard.py --watch"; do
  pid=$(ps -eo pid,args | grep -F "$pat" | grep -v grep | awk '{print $1}')
  [ -n "$pid" ] && { echo "stop old: $pid ($pat)"; kill $pid 2>/dev/null; }
done
sleep 1
(setsid nohup bash setup/auto_pull.sh >> cloud_pull/auto_pull.log 2>&1 &)
(setsid nohup python3 ops/dashboard.py --watch 120 >> cloud_pull/dashboard.log 2>&1 &)
sleep 2
echo "started: $(ps -eo args | grep -cE '[a]uto_pull.sh|[d]ashboard.py --watch') 个进程"
echo "看板: ops/dashboard.html  (Windows: \\\\wsl\$\\Ubuntu\\home\\zechen\\cot-compress\\ops\\dashboard.html)"
