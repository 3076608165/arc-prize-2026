> 2026-09-11 更新：AGI-2 两组已完成，截止前覆盖 32→40，最终正确输出 4/172→13/172；这是单组公开限时诊断。完整比赛 augment8 v10 已出分 27.64，未确立提升。比赛实际公开数据是 172 个输出，下文历史 167 属于旧 Git 数据口径。AGI-3 活跃目录已替换为 agi3-coverage-v2-offline，旧 v1 失败后归档；新轮正在运行。详见 next-direction-20260911.md。

# AGI-2 / AGI-3 本轮交付

本轮完成代码、可复现打包、实验协议和本地验证；执行权限开放后，AGI-2 两个私有限时对照与 AGI-3 冻结配对实验均已启动，尚未取得完整性能实验结果。以下候选均未被确立为提升。构建产物的哈希和逐包检查结果见 `coverage-delivery-verification.json`。

## AGI-2

继续检验减少 TTT 增广是否能提高限时覆盖率。沿用注明来源的 NVARC 模型与求解器，比较 16 次和 8 次增广。

- `agi2/out/coverage-benchmark16`、`coverage-benchmark8`：公开 120 题、167 个输出，每组 1 小时，共同的排序队列、worker hash seed 0、上游固定镜像和 L4 配置。使用两个独立私有 Notebook ID，均已上传为第 1 版并确认 RUNNING；拉回的源码与本地逐单元一致。
- `agi2/out/audited16`、`audited8`：11.83 小时的完整比赛配置，附覆盖审计。普通 Notebook 运行保留上游四题快捷检查；比赛重跑才排入全部隐藏题。普通四题检查不能测量覆盖率。
- 日志记录排队、开始、训练、解码、候选写入、截断与异常。填充答案不计入候选覆盖；“有候选”也不代表答对。
- `agi2/compare_coverage.py` 分别报告覆盖输出数、全部公开输出正确率，以及双方均产生候选的输出上的得失。

公开限时实验仍是诊断：单次运行、排序队列以及公开/隐藏题差异都限制结论；相同 hash seed 也不保证 GPU 数值完全确定。若双方都提前跑完，本次实验仍不能证明限时覆盖优势。审计本身也有少量运行开销。

已有 v10 提交 56140934 在最后保存的服务器快照（2026-09-10 11:57 UTC）仍为 PENDING，不能将旧成绩填作它的成绩。

## AGI-3

- `agi3/out/stable-control`：仅统一 SHA256 种子的基线。
- `agi3/out/coverage-candidate`：同样的种子规则，最多每十次点击抽一次可见物体，其余推进完整像素扫描，不永久排除曾无效的坐标。在不切换关卡、持续调用点击策略的条件下，4552 次点击覆盖全部 4096 像素；这不保证解题。
- `agi3-coverage-v1/protocol.json` 冻结 12 组种子、25 个公开游戏、每游戏最多 5000 个动作、交替执行顺序、源码和运行环境哈希。
- 晋级线上实验须同时满足：12 组完整配对，均值增益至少为对照的 15% 且至少 0.02 分，配对 bootstrap 95% 区间下限大于零，双侧 sign-flip p≤0.05，至少 8 组正增益。该门槛是事先选定的决策规则；通过也不等于隐藏集提升。

结果不完整或源码/环境变化时不能晋级。不满足门槛就保留“未确立”，不看完结果后追加种子。12 组是固定计算预算，不能保证统计功效；区间只描述这些固定公开游戏下的策略随机性。

先前 objects 的三种子 +8.1% 仍为未确立；+43% 仅来自一个种子。状态文件已用带来源限制的历史分布替代 AGI-3 的 `best_public_score`，0.01–0.18 的波动不能被忽略。现有 objects 的 0.16 也只是一次排行榜观测。

## 复现入口

在仓库根目录运行：

```powershell
python agi2/prepare_coverage_bundles.py
python scripts/verify_coverage_bundles.py
python agi2/test_coverage_audit.py
python agi2/test_compare_coverage.py
python agi3/test_paired_analysis.py
```

AGI-2 两个公开诊断运行结束并下载各自输出后：

```powershell
python agi2/compare_coverage.py <16次增广输出目录> <8次增广输出目录> --out experiments/agi2-coverage-results.json
```

AGI-3 的冻结实验入口（需要 starter 的已安装运行环境和执行权限）：

```powershell
& '.\data\ARC-AGI-3-Kaggle-Starter\.venv\Scripts\python.exe' agi3/paired_experiment.py run
```

脚本可从已保存的完整运行继续，不自动提交 Kaggle。不要重建或修改已有协议来绕过哈希检查。

已验证：AGI-2 审计 3 项、覆盖比较 2 项、AGI-3 纯分析 5 项、提交脚本 2 项测试通过；此前本轮在 starter 环境中通过 AGI-3 配对机制 4 项与原有实验 3 项测试，并对两个真实游戏完成各 20 动作的预算冒烟检查。六包的嵌入 Python 已编译检查；没有验证完整 Notebook 的 shell/magic 执行或新 GPU 运行。

此前 AGI-3 启动曾被审批额度限制拒绝。用户开放执行权限后已成功启动冻结实验；AGI-2 两组新诊断也已成功上传并确认 RUNNING。采集程序会下载并分析输出，本任务每 10 分钟继续检查，状态无变化时保持安静。实时状态见两个实验目录；运行完成前不能宣称已测增益。本轮没有新增比赛提交。
