# MCTS 详解：蒙特卡洛树搜索

> 本项目的 MCTS 实现文档，覆盖原理、代码结构、关键优化与配置。

## 1. 为什么需要 MCTS

围棋局面空间约 $10^{170}$，暴力搜索不可行。两种主流思路：

| 思路 | 代表 | 特点 |
|------|------|------|
| **固定深度搜索** | Minimax + αβ剪枝 | 深度/宽度固定，资源均摊 |
| **蒙特卡洛树搜索** | AlphaGo / KataGo | 深度动态，资源**聚焦**关键分支 |

MCTS 的核心洞察：**不是所有分支都值得同样计算量**。用统计（访问次数）+ 先验（策略网络）引导搜索，把算力集中到最有希望的子树上。

---

## 2. 核心思想：四步循环

```
     ① 选择 Selection
        │
     ④ 回溯 ←─────┐  ② 扩展 Expansion
   Backpropagation │
                   │  ③ 评估 Evaluation
```

### ① 选择（Selection）

从根节点出发，沿 **UCB 公式**逐层选子，直到叶节点：

$$
\text{UCB}(child) = Q + c \cdot P \cdot \sqrt{\frac{\ln N_{\text{parent}}}{N_{\text{child}}}}
$$

- $Q$：子节点平均价值（`total_value / visit_count`）——利用
- $P$：策略网络先验概率——引导
- $\sqrt{\ln N_{parent} / N_{child}}$：访问少的分支奖励大——探索
- $c$：UCB 常数（`mcts_c_puct`），平衡探索/利用

**未访问子节点 UCB = +∞**（本项目实现）——保证所有子节点先被探索一遍，避免高先验节点垄断。

### ② 扩展（Expansion）

到达叶节点后，用策略网络 `predict_full_probs` 得到 362 维概率作为先验，展开前 `mcts_expand_width` 个合法子节点。

### ③ 评估（Evaluation）

叶节点的局面价值，两种方式（`use_own_value_net` 决定）：

| 方式 | 评估器 | 特点 |
|------|--------|------|
| `true` | 自训练价值网络 | 快，GPU 批量 |
| `false` | KataGo | 准，慢 |

### ④ 回溯（Backpropagation）

将叶节点价值沿路径向上传播，更新每个节点的 `visit_count += 1`、`total_value += value`。对手视角价值取反（$1 - value$）。

---

## 3. 本项目实现：`src/game/mcts.py`

### 3.1 数据结构

```
MCTSNode:
    board          # sgfmill Board
    color          # 该轮到谁走 'b'/'w'
    move           # 从父到本节点的落子
    prior          # 策略网络先验概率 P
    visit_count    # 访问次数 N
    total_value    # 累计价值（root_player 视角）
    children       # 子节点列表
    is_terminal    # 终局
    expanded       # 是否已扩展
```

### 3.2 核心方法

| 方法 | 作用 |
|------|------|
| `_select_leaf()` | 沿 UCB 到叶节点 + 虚拟访问 |
| `_expand()` | 策略网络先验展开子节点 |
| `_evaluate()` | 单节点评估（价值网络/KataGo） |
| `_evaluate_batch()` | **批量评估 + 回溯**（叶节点并行） |
| `_backprop()` | 沿路径更新价值 |
| `_select_best()` | 访问次数最多的子节点落子 |
| `search()` | 主循环 |

---

## 4. 关键优化：叶节点并行批量

MCTS 串行结构下，每次模拟只评估 1 个叶节点。用 KataGo 时 100 次模拟 = 100 次单查（每次 ~0.2s = 20s）。

**优化**：一批并行路径 + 批量评估。

```
改前（串行）:
  100 次模拟 → 100 次 KataGo 单查 → 20s

改后（叶节点并行）:
  一批 eval_batch_size 条路径（虚拟访问防重选）
  → 积累 eval_batch_size 个叶节点
  → 1 次 get_value_batch 批量查询
  → 统一回溯
  → 100 次模拟, batch=10 → 仅 10 次查询 → 4s
```

**虚拟访问**：选择路径时先 `visit_count += 1`，防止并行批次内重复选择同一节点。回溯时只加 `total_value`。

---

## 5. 缓存

KataGo 评估缓存 `moves 序列 → 胜率`。MCTS 树中不同落子顺序常到达相同局面，缓存去重避免重复查询。

---

## 6. 完整流程图

```
search_move(steps, board, color, curstep)
  └─ MCTS.search()
       for i in range(mcts_simulations):
           ① _select_leaf()    # UCB 到叶节点 + 虚拟访问
           ② _expand()         # 策略网络先验 → 子节点
           ③ 积累 pending 叶节点
           ④ 批量评估（积累够 batch 或最后一轮）:
                _evaluate_batch()   # get_value_batch / 价值网络批量
                → _backprop()       # 更新路径
           ⑤ 可视化快照 + 步进（可选）
       _select_best()  # 访问次数最多 → 落子
```

---

## 7. 配置参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `search_algorithm` | mcts | `minimax` / `mcts` |
| `mcts_simulations` | 100 | 每次走棋模拟次数 |
| `mcts_c_puct` | 1.4 | UCB 常数 |
| `mcts_expand_width` | 10 | 每节点最大分支数 |
| `mcts_eval_batch_size` | 10 | 批量评估批次（KataGo 加速） |
| `mcts_temperature` | 0.0 | 落子温度（0=贪心） |
| `mcts_visualize` | false | GUI 可视化搜索过程 |
| `mcts_vis_interval` | 0.5 | 可视化步进间隔（秒） |
| `mcts_verbose` | false | 控制台打印搜索过程 |
| `use_own_value_net` | false | true=价值网络，false=KataGo |

---

## 8. 可视化（GUI）

配置 `mcts_visualize: true` 时：

- **棋盘热力图**：候选位置圆点，大小 ∝ 访问次数，颜色 = 价值（红低→绿高），标注访问数
- **右侧树图**：搜索树（深度 ≤ 3），节点大小 ∝ log(访问数)，颜色 = 价值
- `mcts_vis_interval` 控制每步模拟间隔，便于观察搜索收敛过程
- `mcts_verbose` 打印每轮模拟的路径/扩展/评估

---

## 9. MCTS vs BatchMinimax 对比

| 维度 | BatchMinimaxMCR | MCTS |
|------|----------------|------|
| 深度 | 固定 `max_search_depth` | 动态增长 |
| 宽度 | 固定 `top_k` | 最多 `mcts_expand_width` |
| 资源分配 | 均摊所有分支 | UCB 动态聚焦 |
| 评估次数 | 每层全评估 | 每模拟 1 叶节点（可批量） |
| 终止 | 必须完整建树 | 可随时停止 |
| 适用 | 深度浅、确定性耗时 | 深度深、计算聚焦 |

---

## 10. 局限与改进方向

| 局限 | 改进 |
|------|------|
| UCB 先验依赖策略网络质量 | 蒸馏更强策略网络 |
| 批量评估延迟 | 调大 `eval_batch_size`（注意批次延迟） |
| 无 PUCT 自适应 | 尝试不同 `c_puct` 值 |
| 温度恒 0（贪心） | 对局早期设温度鼓励探索 |
| 终局判定简化（数字） | 接 KataGo 判定或死活判断 |
