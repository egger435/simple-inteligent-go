# AI 棋力评估：基于 ELO 分数的客观评测方案

## 1. 为什么需要基准对手

ELO 是**相对分数**——单独评估一个 AI 没有意义，必须与已知棋力的对手对战，由胜率推导分数。

$$
\text{期望胜率} \quad E = \frac{1}{1 + 10^{(R_{\text{对手}} - R_{\text{AI}})/400}}
$$

- $R_{\text{AI}}$：AI 的 ELO 分
- $R_{\text{对手}}$：基准对手的 ELO 分
- $E$：AI 对基准对手的期望胜率

由实际胜率反推分数差：

$$
\Delta R = 400 \times \log_{10}\left(\frac{E}{1 - E}\right)
$$

**例**：AI 对 1800 分对手胜率 60%：

$$
\Delta R = 400 \times \log_{10}\left(\frac{0.6}{0.4}\right) \approx 400 \times 0.176 = 70
$$

$$
R_{\text{AI}} = 1800 + 70 = 1870
$$

---

## 2. 基准对手：KataGo GTP

KataGo 支持 GTP 模式，用 `-override-config maxVisits=N` 控制棋力（visits 越多越强）。

| visits 档位 | 大致棋力 | 参考 ELO |
|-------------|---------|---------|
| 1 | 弱 | ~800 |
| 10 | 中 | ~1300 |
| 100 | 强 | ~1800 |

> 注：ELO 为社区经验值，随模型版本浮动。真实评测建议多次采样取中位数。

---

## 3. 评测脚本：`eval_elo.py`

```
你的 AI vs KataGo GTP (visits=1, 10, 100)

对每个档位:
  对 10 局（黑白轮换）
  → 统计胜率 E
  → ELO 公式推导 ΔR
  → 结合锚点 ELO → 得到 AI 分数估计
```

**评测四种模式**：

| 模式 | 算法 | 评估 |
|------|------|------|
| `A_value` | Minimax | 自训练价值网络 |
| `B_kata` | Minimax | KataGo |
| `C_mcts` | MCTS | 自训练价值网络 |
| `D_mcts_kata` | MCTS | KataGo |

运行：

```bash
python scripts/eval_elo.py
```

---

## 4. 对战测试：`test_battle.py`

不需要 GTP 基准，直接让两种模式互搏（MCTS+KataGo vs Minimax+KataGo 等）。

```bash
# 模式: A=Minimax+价值网络 B=Minimax+KataGo
#       C=MCTS+价值网络  D=MCTS+KataGo
python scripts/test_battle.py --left D --right B    # MCTS+KataGo vs Minimax+KataGo
python scripts/test_battle.py --games 5 --steps 60
```

---

## 5. 关键注意事项

| 事项 | 说明 |
|------|------|
| 局数要够 | 每档 ≥ 10 局，胜率误差才可控 |
| 黑白轮换 | 固定执黑/执白会偏（贴目 7.5） |
| 终局判定 | KataGo 判定胜负 |
| 单进程限制 | `eval_elo.py` 逐档串行启动 GTP KataGo |
| 棋力区间 | 搜索参数不同 → 输出区间而非单点 |

---

## 6. 局限与改进

| 局限 | 改进方向 |
|------|---------|
| 锚点 ELO 是近似值 | 与已知强 AI 对局校准 |
| 每档胜率样本有限 | 增加局数；或使用 SPRT 序贯检验提前终止 |
| 只测单一配置 | 分别测不同 top_k / depth / mcts 参数 |
| 耗时较长 | MCTS+KataGo 最慢，可用小 batch 或降模拟数粗测 |

---

## 7. 参考

- ELO 评分系统：国际象棋标准，广泛用于围棋引擎对比
- KataGo GTP `maxVisits`：官方支持的对局强度参数
