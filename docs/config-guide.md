# 配置文件说明

`config.json` 是项目的中心配置，所有可调参数集中管理。本文件逐项解释每个参数的含义、取值范围和建议。

> **用法**：复制 `config.example.json` 为 `config.json` 后按需修改。JSON 不支持注释，本文件是配置项的说明文档。

## 核心设置

| 参数 | 默认 | 说明 |
|------|------|------|
| `device` | `"cuda"` | 运行设备。`"cuda"` 用 GPU（快），`"cpu"` 用 CPU（慢但兼容） |
| `board_size` | `19` | 棋盘大小。标准 19×19，改小（如 9）可加速调试 |
| `pass_label` | `361` | 弃行在 362 维输出中的索引（0~360 是落子，361 是弃行） |
| `have_i` | `true` | 坐标系统是否包含 I 列。`true` = 含 I（A-T），`false` = 不含 I（A-S，跳过 I） |
| `color_map` | `{"1": 0.0, "2": 1.0}` | 行棋方数值映射（1=黑，2=白），用于网络输入通道 |

## 模型路径

| 参数 | 默认 | 说明 |
|------|------|------|
| `strategy_model_path` | `output_models/go_strategy_model_1_1.pth` | 策略网络权重路径 |
| `strategy_model_type` | `"GoCNN_p"` | 策略网络结构：`GoCNN` / `GoCNN_p` / `GoCNN_t` / `AlphaCNN` |
| `value_model_path` | `output_models/go_final_val_model_1_2.pth` | 价值网络权重路径 |

## KataGo 引擎

| 参数 | 默认 | 说明 |
|------|------|------|
| `kata_exe_path` | `katago/katago.exe` | KataGo 可执行文件路径（自行下载） |
| `kata_model_path` | `katago/kata1-*.bin.gz` | KataGo 神经网络权重 |
| `kata_config_path` | `katago/analysis_config.cfg` | KataGo 配置（visits、搜索线程等） |

## 搜索算法

### 算法选择

| 参数 | 默认 | 说明 |
|------|------|------|
| `search_algorithm` | `"mcts"` | 搜索算法：`"minimax"`（固定深度批量搜索）/ `"mcts"`（蒙特卡洛树搜索） |

### Minimax 参数（`search_algorithm="minimax"` 时）

| 参数 | 默认 | 说明 |
|------|------|------|
| `top_k` | `4` | 每个节点展开的分支数。越大搜索越宽但越慢 |
| `max_search_depth` | `5` | 搜索深度。叶节点数 = top_k^depth，深度+1 计算量指数增长 |
| `mc_rollouts` | `5` | 每个叶节点 MC 推演次数 |
| `mc_steps` | `5` | 每次推演走的步数 |

### MCTS 参数（`search_algorithm="mcts"` 时）

| 参数 | 默认 | 说明 |
|------|------|------|
| `mcts_simulations` | `100` | 每次走棋的模拟次数。越多越准但越慢（100≈1-3s） |
| `mcts_c_puct` | `1.4` | UCB 探索常数。越大越偏向探索，越小越偏向利用（参考 AlphaGo Zero） |
| `mcts_expand_width` | `10` | 每个节点最多展开的子节点数。小=窄而深，大=宽而浅 |
| `mcts_eval_batch_size` | `10` | KataGo 批量评估批次。越大 KataGo 往返越少（100 模拟 batch=10 → 10 次查询） |
| `mcts_temperature` | `0.0` | 落子温度。`0` = 贪心选访问最多，`>0` = 按访问次数加权随机（鼓励探索） |

### 评估方式（两种算法共用）

| 参数 | 默认 | 说明 |
|------|------|------|
| `use_own_value_net` | `false` | `true` = 用自训练价值网络评估（快，GPU 批量）；`false` = 用 KataGo（准，慢） |

### MCTS 可视化 / 调试

| 参数 | 默认 | 说明 |
|------|------|------|
| `mcts_visualize` | `false` | GUI 模式是否可视化 MCTS 搜索过程（棋盘热力图 + 树图） |
| `mcts_vis_interval` | `0.5` | 可视化每步模拟间隔（秒），越大看得越清楚但越慢 |
| `mcts_verbose` | `false` | 控制台是否打印搜索过程（选中路径/扩展/评估） |

## 快速调参指南

| 目标 | 调整 |
|------|------|
| **棋力优先** | `mcts_simulations` 提到 200-500，`use_own_value_net: false`（KataGo） |
| **速度优先** | `mcts_simulations` 降到 30-50，`use_own_value_net: true`（价值网络） |
| **看搜索过程** | `mcts_visualize: true` + `mcts_verbose: true` |
| **低配机器** | `device: "cpu"`，`mcts_simulations: 30`，`top_k: 3`，`max_search_depth: 3` |
| **对比两算法** | 切换 `search_algorithm` 后跑 `scripts/test_battle.py` |
| **测棋力** | 跑 `scripts/eval_elo.py`（基于 KataGo visits 档位） |
