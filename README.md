# Kata-EgGO — 简单智能围棋 AI

> 寒假闲着没事，开始学习围棋，结果被各路棋手暴打。于是打算借助科技的力量看看能不能将失去的都夺回来！

基于 **批量并行 MiniMax 搜索** 与 **蒙特卡洛树搜索（MCTS）** 的围棋 AI，支持两种搜索算法和两种局面评估方式（自训练价值网络 / KataGo），带 GUI 可视化与 ELO 棋力评测。

## 功能特性

- **双搜索算法**（`config.json` 切换）：
  - `BatchMinimaxMCR`：固定深度批量并行搜索 + MC 推演
  - `MCTS`：先验 UCB + 价值网络（AlphaGo Zero 风格），叶节点并行批量评估
- **双局面评估**（`use_own_value_net` 切换）：
  - 自训练价值网络（快，GPU 批量）
  - KataGo（准，流水线批量查询）
- **GUI 图形界面**：鼠标落子、概率热力图、胜率走势图、MCTS 搜索过程可视化
- **ELO 棋力评测**：基于 KataGo GTP visits 档位的客观棋力评估
- **对战测试**：任意两种模式互搏

## 组成

- 自训练的卷积策略选择网络（`GoCNN_p`）
- 自训练的卷积价值网络
- 双搜索算法（BatchMinimax / MCTS）
- KataGo 局面价值判断（可选）

## 目录结构

```
src/
  config.py           # 中心配置（JSON 文件 / 环境变量 / CLI）
  common.py           # 公共工具 + 配置导出
  go_play.py          # 主入口（终端 / GUI）
  engine/             # KataGo 引擎封装（analysis + GTP）
  game/               # 搜索算法 + 对局逻辑
    tree.py           # BatchMinimaxMCR + search_move 调度器
    mcts.py           # MCTS 实现
    player.py         # 对局状态管理
    loops.py          # 终端对局循环
  models/             # 策略网络 + 价值网络定义
  strategy/           # 策略网络推理
  value/              # 价值网络推理
  ui/                 # GUI（棋盘渲染、overlay、信息面板）
  rl/                 # 强化学习训练管线
  dataset/            # 数据集生成脚本
scripts/              # 训练 / 评测 / 对战脚本
  train_value_net.py    # 价值网络训练
  finetune_strategy.py  # 策略网络微调
  distill_strategy.py   # KataGo 知识蒸馏
  selftune_value_net.py # 价值网络数据回炉
  rl_train.py           # 强化学习训练
  test_battle.py        # 对战测试
  eval_elo.py           # ELO 棋力评测
docs/                 # 项目文档
```

## 数据集

- 策略网络监督训练：1980-2018 年约 7 万局人类对局 + 约 10 万局 AI 对局
- 数据源：`data/` 下 SGF 棋谱（不随仓库提交，体积 12GB+）

## 策略网络构成

试验了四种结构：简易三层卷积 `GoCNN`、增强全局特征 `GoCNN_p`、Transformer 卷积 `GoCNN_t`、AlphaGo 原版 `AlphaCNN`。`GoCNN_p` 速度与准确率最平衡。

`GoCNN_p`：输入 2 维 → 64 维 → 3 个残差块 → 全局平均池化提取全局特征 → 与局部特征融合 → 全连接 → 362 种行棋概率。

## 安装

1. 安装依赖：

   ```bash
   pip install torch numpy sgfmill matplotlib
   ```
2. 准备模型文件（不随仓库提交）：

   - **策略网络**：`output_models/go_strategy_model_1_1.pth`（或自行训练）
   - **价值网络**：`output_models/go_final_val_model_1_2.pth`（或用 `train_value_net.py` 训练）
   - **KataGo**：从 [KataGo Releases](https://github.com/lightvector/KataGo/releases) 下载 `katago.exe` 和模型，放入 `katago/`
3. 复制配置模板：

   ```bash
   cp config.example.json config.json
   # 按需修改模型路径等
   ```

## 使用

```bash
# 终端对局（AI 执白）
python src/go_play.py --ai-color w

# GUI 对局
python src/go_play.py --gui

# 对战测试（MCTS+KataGo vs Minimax+KataGo）
python scripts/test_battle.py --left D --right B

# ELO 评测
python scripts/eval_elo.py

# 价值网络训练
python scripts/train_value_net.py --data data/xxx.npz
```

## 配置说明

完整配置项说明见 [配置文档](docs/config-guide.md)。

关键配置项（`config.json`）：


| 参数                              | 默认  | 说明                        |
| ----------------------------------- | ------- | ----------------------------- |
| `search_algorithm`                | mcts  | `minimax` / `mcts`          |
| `use_own_value_net`               | false | true=价值网络，false=KataGo |
| `top_k` / `max_search_depth`      | 4 / 5 | Minimax 分支数 / 深度       |
| `mcts_simulations`                | 100   | MCTS 模拟次数               |
| `mcts_c_puct`                     | 1.4   | MCTS UCB 常数               |
| `mcts_expand_width`               | 10    | MCTS 每节点最大分支         |
| `mcts_eval_batch_size`            | 10    | KataGo 批量评估批次         |
| `mcts_visualize` / `mcts_verbose` | false | GUI 可视化 / 控制台打印     |


个人学习研究项目。
