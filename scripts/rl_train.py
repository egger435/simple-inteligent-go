'''自对弈强化学习训练入口。

用法::

    python rl_train.py --device cuda --generations 20
    python rl_train.py --device cpu --generations 5 --board-size 9
'''

import sys
import os
import time
import argparse

# ---- 路径处理 ----
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

# ---- CLI ----
parser = argparse.ArgumentParser(
    description='Kata-EgGO: 自对弈强化学习训练',
)
parser.add_argument('--device', '-d', choices=['cpu', 'cuda'], default=None,
                    help='训练设备 (默认: cuda)')
parser.add_argument('--generations', '-gen', type=int, default=None,
                    help=f'训练代数')
parser.add_argument('--games', '-ng', type=int, default=None,
                    help='每代自对弈局数')
parser.add_argument('--board-size', '-bs', type=int, default=None,
                    help='棋盘大小 (9 或 19, 默认: 19)')
args = parser.parse_args()

# ---- 导入 common 并应用参数 ----
import common
if args.device:
    common.config.device = args.device
if args.generations:
    common.config.rl_max_generations = args.generations
if args.games:
    common.config.rl_games_per_generation = args.games
if args.board_size:
    common.config.board_size = args.board_size
common._sync_module_vars()

from common import (
    DEVICE,
    BOARD_SIZE,
    RL_GAMES_PER_GENERATION, RL_BUFFER_CAPACITY,
    RL_BATCH_SIZE, RL_LEARNING_RATE, RL_TRAIN_STEPS_PER_GEN,
    RL_TEMPERATURE_EARLY, RL_TEMPERATURE_LATE,
    RL_EVAL_GAMES, RL_ACCEPT_THRESHOLD, RL_MAX_GENERATIONS,
    STRATEGY_MODEL_PATH,
)
from rl.self_play import SelfPlayEngine
from rl.replay_buffer import ReplayBuffer
from rl.trainer import RLTrainer
from rl.arena import Arena


# =====================================================================
def main():
    print('=' * 55)
    print('Kata-EgGO: 自对弈强化学习训练 (Phase 1)')
    print('=' * 55)
    print(f'  设备: {DEVICE}')
    print(f'  棋盘: {BOARD_SIZE}×{BOARD_SIZE}')
    print(f'  初始模型: {STRATEGY_MODEL_PATH}')
    print(f'  每代局数: {RL_GAMES_PER_GENERATION}')
    print(f'  最大代数: {RL_MAX_GENERATIONS}')
    print(f'  接受阈值: {RL_ACCEPT_THRESHOLD:.0%}')
    print()

    # ---- 初始化 ----
    engine = SelfPlayEngine(
        temperature_early=RL_TEMPERATURE_EARLY,
        temperature_late=RL_TEMPERATURE_LATE,
    )
    buffer = ReplayBuffer(capacity=RL_BUFFER_CAPACITY)
    trainer = RLTrainer(device=DEVICE, lr=RL_LEARNING_RATE)

    # 保存初始模型（用于 Arena 比对）
    old_state = trainer.get_model_state()
    best_state = old_state

    ckpt_dir = 'output_models'
    rl_ckpt_path = os.path.join(ckpt_dir, 'go_strategy_rl.pth')

    best_winrate = 0.5

    # =================================================================
    # 训练循环
    # =================================================================
    for gen in range(1, RL_MAX_GENERATIONS + 1):
        t_start = time.time()
        print(f'\n{"=" * 50}')
        print(f'  Generation {gen}/{RL_MAX_GENERATIONS}')
        print(f'{"=" * 50}')

        # ---- 自对弈 ----
        print(f'[自对弈] 生成 {RL_GAMES_PER_GENERATION} 局...')
        total_steps = 0
        b_wins = 0
        for i in range(RL_GAMES_PER_GENERATION):
            records, winner, (b_stones, w_stones, n_steps, score_lead) = engine.play_one_game()
            if winner == 'b':
                b_wins += 1
            for board_np, move_idx, reward in records:
                buffer.push(board_np, move_idx, reward)
                total_steps += 1

            # 每局打印进度
            pct = (i + 1) / RL_GAMES_PER_GENERATION
            score_str = f'B+{score_lead:.1f}' if score_lead and score_lead > 0 else f'W+{-score_lead:.1f}' if score_lead else '?'
            sys.stdout.write(
                f'\r  局 {i+1}/{RL_GAMES_PER_GENERATION} '
                f'({pct:.0%}) | 缓冲池 {len(buffer):,} 步 | '
                f'黑胜率 {b_wins/(i+1):.1%} | '
                f'{score_str} ({n_steps}手)'
            )
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        print(f'  完成: {total_steps:,} 步入池 | '
              f'黑胜率 {b_wins/RL_GAMES_PER_GENERATION:.1%}')

        # ---- 训练 ----
        print(f'[训练] {RL_TRAIN_STEPS_PER_GEN} 步策略梯度更新...')
        trainer.train()
        total_loss = 0.0
        for step in range(RL_TRAIN_STEPS_PER_GEN):
            boards, moves, rewards = buffer.sample(RL_BATCH_SIZE)
            result = trainer.train_step(boards, moves, rewards)
            total_loss += result['loss']

            if (step + 1) % 100 == 0 or step == RL_TRAIN_STEPS_PER_GEN - 1:
                avg_loss = total_loss / (step + 1)
                pct = (step + 1) / RL_TRAIN_STEPS_PER_GEN
                sys.stdout.write(
                    f'\r  步 {step+1}/{RL_TRAIN_STEPS_PER_GEN} '
                    f'({pct:.0%}) | loss: {avg_loss:.4f} | '
                    f'reward: {result["mean_reward"]:+.4f}'
                )
                sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        avg_loss = total_loss / RL_TRAIN_STEPS_PER_GEN
        print(f'  完成: avg_loss={avg_loss:.4f}')

        # ---- 评估 ----
        print(f'[评估] 新模型 vs 旧模型 ({RL_EVAL_GAMES} 局)...')
        trainer.eval()
        new_state = trainer.get_model_state()

        # 用 Arena 评估（纯策略网络贪心对弈，无搜索）
        new_winrate = Arena(
            _make_model_from_state(new_state, DEVICE),
            _make_model_from_state(old_state, DEVICE),
        ).evaluate(RL_EVAL_GAMES)

        # ---- 决策 ----
        if new_winrate >= RL_ACCEPT_THRESHOLD:
            print(f'✅ 新模型通过！胜率 {new_winrate:.1%} ≥ {RL_ACCEPT_THRESHOLD:.0%}')
            old_state = new_state
            trainer.save_checkpoint(rl_ckpt_path)
            if new_winrate > best_winrate:
                best_winrate = new_winrate
                best_state = new_state
                best_path = os.path.join(ckpt_dir, 'go_strategy_rl_best.pth')
                trainer.save_checkpoint(best_path)
                print(f'  最佳模型已保存: {best_path}')
        else:
            print(f'❌ 新模型未通过。胜率 {new_winrate:.1%} < {RL_ACCEPT_THRESHOLD:.0%}')
            print(f'  回退到旧模型')
            trainer.load_model_state(old_state)

        elapsed = time.time() - t_start
        print(f'  耗时: {elapsed:.0f}s | 最佳胜率: {best_winrate:.1%}')

    # =================================================================
    print(f'\n{"=" * 50}')
    print(f'训练完成。最佳模型胜率: {best_winrate:.1%}')
    print(f'最终模型: {rl_ckpt_path}')
    print(f'最佳模型: {os.path.join(ckpt_dir, "go_strategy_rl_best.pth")}')


# =====================================================================
def _make_model_from_state(state_dict: dict, device: str):
    '''从参数状态字典创建一个模型实例（复用 Config 的工厂方法）。'''
    from config import config
    model = config.create_strategy_model()
    model.load_state_dict(state_dict)
    return model.to(device)


# =====================================================================
if __name__ == '__main__':
    main()
