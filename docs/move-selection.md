# AI 落子选择流程

## 总览

AI 走一步棋由统一调度器 `search_move` 按配置选择两种算法之一：

```
search_move(steps, board, color, curstep)
  ├─ config.search_algorithm == 'minimax' → BatchMinimaxMCR
  └─ config.search_algorithm == 'mcts'    → MCTS
```

两种算法都输出 `(best_move, best_value)`，接入点（终端对局 / GUI / 对战测试 / ELO 评测）无需区分。

---

# 算法一：BatchMinimaxMCR（固定深度批量搜索）

[src/game/tree.py](src/game/tree.py) — 固定深度 BFS 全展开 + 批量评估。

## 建树（`build_tree`）

```
深度 0:  1 节点   → top_k 子节点
深度 1:  top_k    → top_k² 子节点
...
深度 D:  top_k^D → 叶节点
```

每层**一次 `predict_full_batch`** 批量策略网络推理，替代逐节点调用。

## 评估叶节点（`evaluate_all_leaves`）

根据 `use_own_value_net` 双路径：

| 开关 | 评估方式 |
|------|---------|
| `true` | 每个叶节点复制 `mc_rollouts` 份 → 贪心推演 `mc_steps` 步 → 批量价值网络评估 → 取平均 |
| `false` | 收集叶节点落子序列 → `get_value_batch` 流水线发 KataGo |

## 回溯 + 选着

- MiniMax 回溯：我方 max / 对方 min
- 选 value 最大的子节点

**特点**：深度/宽度固定，每分支计算均摊，GPU 调用次数恒定。

---

# 算法二：MCTS（蒙特卡洛树搜索）

[src/game/mcts.py](src/game/mcts.py) — 先验 UCB + 价值网络（AlphaGo Zero 风格）。

## 四步循环

```
MCTS.search():
  ① 选择 Selection:   沿 UCB 到叶节点
     UCB = Q + c·P·√(ln N_parent / N_child)
     未访问子节点 UCB=+inf（先全探索一遍）
  ② 扩展 Expansion:   策略网络 predict_full_probs → 先验 P → 建子节点
  ③ 评估 Evaluation:  叶节点价值 → 价值网络 或 KataGo
  ④ 回溯 Backprop:    沿路径更新 visit_count + total_value
  落子:                访问次数最多的子节点
```

## 叶节点并行批量（加速 KataGo）

```
一批 eval_batch_size 条并行路径
  → 虚拟访问（visit_count+1 防重复选择）
  → 积累 eval_batch_size 个叶节点
  → 1 次 get_value_batch 批量评估
  → 统一回溯

例: 100 模拟, batch=10 → 仅 10 次 KataGo 查询（原 100 次）
```

## 动态深度

MCTS **无固定深度**——随模拟次数自然增长。`mcts_expand_width` 控制每节点分支数，`mcts_simulations` 控制总计算量。

## 可视化（GUI 模式）

配置 `mcts_visualize: true` 时：
- **棋盘热力图**：候选位置圆点，大小 ∝ 访问次数，颜色 = 价值
- **右侧树图**：搜索树，节点大小 ∝ log(访问数)，深度 ≤ 3
- `mcts_vis_interval` 控制每步模拟间隔
- `mcts_verbose` 控制控制台打印（路径/扩展/评估）

---

# 评估方式（两种算法共用）

`use_own_value_net` 决定叶节点/局面评估：

| 开关 | 评估器 | 速度 | 精度 |
|------|--------|------|------|
| `true` | 自训练价值网络 `go_final_val_model_1_2.pth` | 快（GPU 批量） | 依赖训练质量 |
| `false` | KataGo（`get_value_batch` 流水线） | 慢 | 高 |

---

# 完整调用链

```
go_play.py / ui.app / test_battle.py / eval_elo.py
  → game.tree.search_move()
      ├─ BatchMinimaxMCR.search()
      │    ├─ build_tree()           # 每层 predict_full_batch
      │    ├─ evaluate_all_leaves()  # MC推演 或 KataGo 批量
      │    ├─ minimax_backup()
      │    └─ select_best_move()
      └─ MCTS.search()
           ├─ _select_leaf()         # UCB + 虚拟访问
           ├─ _expand()              # 策略网络先验
           ├─ _evaluate_batch()      # 批量评估 + 回溯
           └─ _select_best()         # 访问次数最多
```

---

# 关键配置（config.json）

| 参数 | 默认 | 说明 |
|------|------|------|
| `search_algorithm` | mcts | `minimax` 或 `mcts` |
| `top_k` | 4 | Minimax 每节点分支数 |
| `max_search_depth` | 5 | Minimax 深度 |
| `mc_rollouts` / `mc_steps` | 5/5 | Minimax MC 推演参数 |
| `use_own_value_net` | false | true=价值网络，false=KataGo |
| `mcts_simulations` | 100 | MCTS 每次走棋模拟次数 |
| `mcts_c_puct` | 1.4 | MCTS UCB 常数 |
| `mcts_expand_width` | 10 | MCTS 每节点最大分支 |
| `mcts_eval_batch_size` | 10 | KataGo 批量评估批次 |
| `mcts_temperature` | 0.0 | 落子温度（0=贪心） |
| `mcts_visualize` | false | GUI 可视化搜索过程 |
| `mcts_vis_interval` | 0.5 | 可视化步进间隔（秒） |
| `mcts_verbose` | false | 控制台打印搜索过程 |

---

# 两算法对比

| 维度 | BatchMinimaxMCR | MCTS |
|------|----------------|------|
| 深度 | 固定 `max_search_depth` | 动态增长 |
| 宽度 | 固定 `top_k` | 最多 `mcts_expand_width` |
| 资源分配 | 均摊所有分支 | UCB 动态聚焦 |
| 评估次数 | 每层全评估 | 每模拟 1 叶节点（可批量） |
| 适用 | 深度浅、确定性耗时 | 深度深、计算聚焦 |
