# Deviations log（预注册之后的全部代码改动）

预注册：OSF https://osf.io/usycb（2026-09-19 02:23:34），引用 freeze r4 `d207c263df938f6003e51ff3da90f13367b40d99`（README 填 hash 的提交 `471f0561`）。
规则：此后任何代码 / 配置改动都记在这里，带 commit hash、原因、是否影响预注册的分析或判定。

| # | 日期 | commit | 改动 | 原因 | 对预注册的影响 |
|---|---|---|---|---|---|
| D1 | 2026-09-19 | `832ec52` | `setup/sync_cloud.sh`：默认目标改为新 pod `root@195.26.233.96:27805`；push 规则放行 `results/whitelist_extra_banned.json` 与 `results/whitelist_audit.md`，并排除 `archive/` | RunPod 自动迁移换了机器（/workspace 迁移后为空，需重推重装）；原 push 规则排除整个 `results/`，会让 pod 缺少 287 个 id 的审计禁集 → 白名单变成 8638 / 8742（与预注册的 8351 / 8455 不符） | 无（基础设施；保证 pod 上的白名单与预注册一致） |
