'''
公共配置与工具函数模块。

从 config.py 导入配置并通过模块级变量暴露，保持对原有 `from common import *`
用法的向后兼容。模型改为懒加载，避免在 import 时就占用 GPU。
'''

import os
import numpy as np
import torch
from sgfmill import sgf, boards

import models.strategy_net as sg
import models.value_net as va
from config import config  # noqa: F401 — 供外部显式导入


# =====================================================================
# 从 Config 导出所有配置项（模块级变量，兼容 `from common import *`）
# =====================================================================

# 核心 ============================
DEVICE: str                          = config.device
BOARD_SIZE: int                      = config.board_size
PASS_LABEL: int                      = config.pass_label
HAVE_I: bool                         = config.have_i
GAME_KOMI: float                     = 7.5
COLOR_MAP: dict                      = config.color_map

# 路径 ============================
SGF_DATASET_DIR: str                 = config.sgf_dataset_dir
SG_DATASET_SAVE_PATH: str            = config.sg_dataset_save_path
SG_CHUNK_SAMPLE_NUM: int             = config.sg_chunk_sample_num
VA_DATASET_SAVE_PATH: str            = config.va_dataset_save_path
LOG_PATH: str                        = config.log_path

# 策略网络（训练）=================
SG_CHUNK_DIR: str                    = config.sg_chunk_dir
SG_VALID_CHUNK_DIR: str              = config.sg_valid_chunk_dir
SG_SAVE_MODEL_PATH: str              = config.sg_save_model_path
SG_BEST_MODEL_PATH: str              = config.sg_best_model_path
SG_VALID_INTERNAL: int               = config.sg_valid_interval
SG_SCHEDULER_PATIENCE: int           = config.sg_scheduler_patience
SG_SCHEDULER_FACTOR: float           = config.sg_scheduler_factor
SG_NUM_GLOBAL_EPOCHS: int            = config.sg_num_global_epochs
SG_BATCH_SIZE: int                   = config.sg_batch_size
SG_EPOCHS_PER_CHUNK: int             = config.sg_epochs_per_chunk
SG_LEARNING_RATE: float              = config.sg_learning_rate
SG_WEIGHT_DECAY: float               = config.sg_weight_decay
SG_GRADIENT_ACCUMULATION_STEPS: int  = config.sg_gradient_accumulation_steps
SG_GRADIENT_CLIP: float              = config.sg_gradient_clip

# 价值网络（训练）=================
VA_DATASET_PATH: str                 = config.va_dataset_path
VA_SAVE_MODEL_PATH: str              = config.va_save_model_path
VA_BATCH_SIZE: int                   = config.va_batch_size
VA_LEARNING_RATE: float              = config.va_learning_rate
VA_WEIGHT_DECAY: float               = config.va_weight_decay
VA_PATIENCE: int                     = config.va_patience

# 策略选择（推理）=================
STRATEGY_MODEL_PATH: str             = config.strategy_model_path
STRATEGY_MODEL_TYPE: str             = config.strategy_model_type
TOP_K: int                           = config.top_k

# 价值判断（推理）=================
VALUE_MODEL_PATH: str                = config.value_model_path

# KataGo ==========================
KATA_EXE_PATH: str                   = config.kata_exe_path
KATA_MODEL_PATH: str                 = config.kata_model_path
KATA_CONFIG_PATH: str                = config.kata_config_path

# 搜索算法 ========================
MAX_SEARCH_DEEPTH: int               = config.max_search_depth
MC_START_THRESHOLD: int              = config.mc_start_threshold
MC_SIMULATIONS: int                  = config.mc_simulations
USE_OWN_VALUE_NET: bool              = config.use_own_value_net
MC_ROLLOUTS: int                     = config.mc_rollouts
MC_STEPS: int                        = config.mc_steps
SEARCH_ALGORITHM: str                = config.search_algorithm
MCTS_SIMULATIONS: int                = config.mcts_simulations
MCTS_C_PUCT: float                   = config.mcts_c_puct
MCTS_EXPAND_WIDTH: int               = config.mcts_expand_width
MCTS_EVAL_BATCH_SIZE: int            = config.mcts_eval_batch_size
MCTS_TEMPERATURE: float              = config.mcts_temperature
MCTS_VISUALIZE: bool                 = config.mcts_visualize
MCTS_VIS_INTERVAL: float             = config.mcts_vis_interval
MCTS_VERBOSE: bool                   = config.mcts_verbose

# 强化学习 =========================
RL_GAMES_PER_GENERATION: int         = config.rl_games_per_generation
RL_BUFFER_CAPACITY: int              = config.rl_buffer_capacity
RL_BATCH_SIZE: int                   = config.rl_batch_size
RL_LEARNING_RATE: float              = config.rl_learning_rate
RL_TRAIN_STEPS_PER_GEN: int          = config.rl_train_steps_per_gen
RL_TEMPERATURE_EARLY: float          = config.rl_temperature_early
RL_TEMPERATURE_LATE: float           = config.rl_temperature_late
RL_EVAL_GAMES: int                   = config.rl_eval_games
RL_ACCEPT_THRESHOLD: float           = config.rl_accept_threshold
RL_MAX_GENERATIONS: int              = config.rl_max_generations

# 自对弈数据标注 =====================
SP_GAMES: int                        = config.sp_games
SP_SAMPLE_INTERVAL: int              = config.sp_sample_interval
SP_TEMPERATURE: float                = config.sp_temperature
SP_KATAGO_VISITS: int                = config.sp_katago_visits
SP_OUTPUT_PATH: str                  = config.sp_output_path


# =====================================================================
# 懒加载模型
# =====================================================================

_strategy_model = None   # 策略网络模型缓存
_value_model = None      # 价值网络模型缓存


def get_strategy_model():
    '''懒加载并返回策略网络模型（首次调用时创建，后续复用缓存）。

    使用 config.device 和 config.strategy_model_path 的设置，
    因此必须在配置更新之后调用。'''
    global _strategy_model
    if _strategy_model is None:
        _strategy_model = config.create_strategy_model()
    return _strategy_model


def _set_strategy_model(model):
    '''替换缓存的策略网络模型（供 RL Arena 使用）。'''
    global _strategy_model
    _strategy_model = model


def get_value_model():
    '''懒加载并返回价值网络模型（首次调用时创建，后续复用缓存）。'''
    global _value_model
    if _value_model is None:
        _value_model = config.create_value_model()
    return _value_model


# =====================================================================
# 运行时配置更新（供 go_play.py 在模型加载前调用）
# =====================================================================

def update_config_from_args(args) -> None:
    '''从 CLI 参数更新 Config 实例和本级模块级变量。

    必须在任何模型（策略网络 / 价值网络 / KataGo）创建之前调用，
    否则设备及路径设置不会生效。
    '''
    config.update_from_args(args)
    _sync_module_vars()


def _sync_module_vars() -> None:
    '''将 config 中的值同步到模块级变量（兼容 `from common import *`）。

    在 update_config_from_args() 和 load_json 重载后调用。
    注意：已经 `from common import *` 的模块，其本地引用不会自动更新；
    因此 go_play.py 需要在其他 ML 模块 import 之前调用本函数。
    '''
    import common as self_module

    # 核心
    self_module.DEVICE             = config.device
    self_module.BOARD_SIZE         = config.board_size
    self_module.PASS_LABEL         = config.pass_label
    self_module.HAVE_I             = config.have_i
    self_module.COLOR_MAP          = config.color_map

    # 路径
    self_module.SGF_DATASET_DIR    = config.sgf_dataset_dir
    self_module.SG_DATASET_SAVE_PATH = config.sg_dataset_save_path
    self_module.SG_CHUNK_SAMPLE_NUM  = config.sg_chunk_sample_num
    self_module.VA_DATASET_SAVE_PATH = config.va_dataset_save_path
    self_module.LOG_PATH           = config.log_path

    # 策略网络（训练）
    self_module.SG_CHUNK_DIR       = config.sg_chunk_dir
    self_module.SG_VALID_CHUNK_DIR = config.sg_valid_chunk_dir
    self_module.SG_SAVE_MODEL_PATH = config.sg_save_model_path
    self_module.SG_BEST_MODEL_PATH = config.sg_best_model_path
    self_module.SG_VALID_INTERNAL  = config.sg_valid_interval
    self_module.SG_SCHEDULER_PATIENCE = config.sg_scheduler_patience
    self_module.SG_SCHEDULER_FACTOR   = config.sg_scheduler_factor
    self_module.SG_NUM_GLOBAL_EPOCHS  = config.sg_num_global_epochs
    self_module.SG_BATCH_SIZE      = config.sg_batch_size
    self_module.SG_EPOCHS_PER_CHUNK  = config.sg_epochs_per_chunk
    self_module.SG_LEARNING_RATE   = config.sg_learning_rate
    self_module.SG_WEIGHT_DECAY    = config.sg_weight_decay
    self_module.SG_GRADIENT_ACCUMULATION_STEPS = config.sg_gradient_accumulation_steps
    self_module.SG_GRADIENT_CLIP   = config.sg_gradient_clip

    # 价值网络（训练）
    self_module.VA_DATASET_PATH    = config.va_dataset_path
    self_module.VA_SAVE_MODEL_PATH = config.va_save_model_path
    self_module.VA_BATCH_SIZE      = config.va_batch_size
    self_module.VA_LEARNING_RATE   = config.va_learning_rate
    self_module.VA_WEIGHT_DECAY    = config.va_weight_decay
    self_module.VA_PATIENCE        = config.va_patience

    # 策略选择（推理）
    self_module.STRATEGY_MODEL_PATH = config.strategy_model_path
    self_module.STRATEGY_MODEL_TYPE = config.strategy_model_type
    self_module.TOP_K              = config.top_k

    # 价值判断（推理）
    self_module.VALUE_MODEL_PATH   = config.value_model_path

    # KataGo
    self_module.KATA_EXE_PATH      = config.kata_exe_path
    self_module.KATA_MODEL_PATH    = config.kata_model_path
    self_module.KATA_CONFIG_PATH   = config.kata_config_path

    # 搜索算法
    self_module.MAX_SEARCH_DEEPTH  = config.max_search_depth
    self_module.MC_START_THRESHOLD = config.mc_start_threshold
    self_module.MC_SIMULATIONS          = config.mc_simulations
    self_module.USE_OWN_VALUE_NET      = config.use_own_value_net
    self_module.MC_ROLLOUTS            = config.mc_rollouts
    self_module.MC_STEPS               = config.mc_steps
    self_module.SEARCH_ALGORITHM       = config.search_algorithm
    self_module.MCTS_SIMULATIONS       = config.mcts_simulations
    self_module.MCTS_C_PUCT            = config.mcts_c_puct
    self_module.MCTS_EXPAND_WIDTH      = config.mcts_expand_width
    self_module.MCTS_EVAL_BATCH_SIZE   = config.mcts_eval_batch_size
    self_module.MCTS_TEMPERATURE       = config.mcts_temperature
    self_module.MCTS_VISUALIZE         = config.mcts_visualize
    self_module.MCTS_VIS_INTERVAL      = config.mcts_vis_interval
    self_module.MCTS_VERBOSE           = config.mcts_verbose

    # 强化学习
    self_module.RL_GAMES_PER_GENERATION  = config.rl_games_per_generation
    self_module.RL_BUFFER_CAPACITY       = config.rl_buffer_capacity
    self_module.RL_BATCH_SIZE            = config.rl_batch_size
    self_module.RL_LEARNING_RATE         = config.rl_learning_rate
    self_module.RL_TRAIN_STEPS_PER_GEN   = config.rl_train_steps_per_gen
    self_module.RL_TEMPERATURE_EARLY     = config.rl_temperature_early
    self_module.RL_TEMPERATURE_LATE      = config.rl_temperature_late
    self_module.RL_EVAL_GAMES            = config.rl_eval_games
    self_module.RL_ACCEPT_THRESHOLD      = config.rl_accept_threshold
    self_module.RL_MAX_GENERATIONS       = config.rl_max_generations

    # 自对弈数据标注
    self_module.SP_GAMES             = config.sp_games
    self_module.SP_SAMPLE_INTERVAL   = config.sp_sample_interval
    self_module.SP_TEMPERATURE       = config.sp_temperature
    self_module.SP_KATAGO_VISITS     = config.sp_katago_visits
    self_module.SP_OUTPUT_PATH       = config.sp_output_path


# =====================================================================
# 工具函数
# =====================================================================

def go_str_to_idx(go_str: str, have_i=None):
    '''将棋盘字符坐标输入转化为索引坐标 (row, col)。

    Args:
        go_str: 坐标字符串，如 "J8"
        have_i: 是否使用包含 I 列的坐标系统；None 时使用全局 HAVE_I
    '''
    if have_i is None:
        have_i = HAVE_I
    col_char = go_str[0].upper()
    row_num = int(go_str[1:])

    col_map_I = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
        'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15,
        'Q': 16, 'R': 17, 'S': 18,
    }
    col_map = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
        'J': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'O': 13, 'P': 14, 'Q': 15,
        'R': 16, 'S': 17, 'T': 18,
    }

    col_idx = col_map_I[col_char] if have_i else col_map[col_char]
    row_idx = row_num - 1
    return (row_idx, col_idx)


def idx_to_go_str(idx, have_i=None):
    '''将索引坐标 (row, col) 转化为棋盘字符坐标。'''
    if have_i is None:
        have_i = HAVE_I
    r, c = idx
    col_map_I_rev = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
        8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
        16: 'Q', 17: 'R', 18: 'S',
    }
    col_map_rev = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
        8: 'J', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q',
        16: 'R', 17: 'S', 18: 'T',
    }
    rownum = r + 1
    colchr = col_map_I_rev[c] if have_i else col_map_rev[c]
    return f'{colchr}{rownum}'


def read_go_sgf(sgf_content: bytes):
    '''根据 SGF 字节流解析对局信息和每步落子记录。

    Returns:
        game_info:    对局信息字典
        step_records: 每步落子记录列表
    '''
    game = sgf.Sgf_game.from_bytes(sgf_content)
    main_sequence = list(game.get_main_sequence())
    board_size = game.get_size()
    black_player = game.get_player_name('b')
    white_player = game.get_player_name('w')
    komi = game.get_komi()
    winner = game.get_winner()

    game_info = {
        'board_size': board_size,
        'black_player': black_player,
        'white_player': white_player,
        'komi': komi,
        'winner': winner,
    }

    board = boards.Board(board_size)
    step_records = []

    for idx, node in enumerate(main_sequence, 1):
        move = node.get_move()
        if not move:
            step_records.append({
                'step': idx,
                'color_code': None,
                'pos': None,
                'pos_str': None,
                'board_matrix': None,
            })
            continue
        color, pos = move
        color_code = 1 if color == 'b' else 2
        pos_str = f'({pos[0]}, {pos[1]})' if pos else 'pass'
        if pos:
            board.play(pos[0], pos[1], color)

        board_matrix = np.zeros((board_size, board_size), dtype=int)
        for r in range(board_size):
            for c in range(board_size):
                stone = board.get(r, c)
                if stone == 'b':
                    board_matrix[r, c] = 1
                elif stone == 'w':
                    board_matrix[r, c] = 2

        step_records.append({
            'step': idx,
            'color_code': color_code,
            'pos': pos,
            'pos_str': pos_str,
            'board_matrix': board_matrix,
        })

    return game_info, step_records


def get_final_from_sgf(sgf_content: bytes):
    '''根据 SGF 字节流得到终局信息，返回 (board_with_komi_matrix, final_val)。'''
    game = sgf.Sgf_game.from_bytes(sgf_content)
    main_sequence = list(game.get_main_sequence())
    winner = game.get_winner()
    komi = game.get_komi()

    board = boards.Board(BOARD_SIZE)
    for idx, node in enumerate(main_sequence, 1):
        move = node.get_move()
        if not move:
            continue
        color, pos = move
        if pos:
            board.play(pos[0], pos[1], color)

    final_board_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            stone = board.get(r, c)
            if stone == 'b':
                final_board_matrix[r, c] = 0.5
            elif stone == 'w':
                final_board_matrix[r, c] = 1.0

    komi_matrix = np.full((BOARD_SIZE, BOARD_SIZE), komi, dtype=np.float32)
    board_with_komi_matrix = np.stack([final_board_matrix, komi_matrix], axis=0)

    if winner == 'b':
        final_val = [1.0, 0.0]
    elif winner == 'w':
        final_val = [0.0, 1.0]
    else:
        final_val = None

    return board_with_komi_matrix, final_val


def get_final_board_from_sgf(sgf_content: bytes):
    '''根据 SGF 字节流读取终局信息，返回 (board, komi, winner)。'''
    game = sgf.Sgf_game.from_bytes(sgf_content)
    main_sequence = list(game.get_main_sequence())
    winner = game.get_winner()
    komi = game.get_komi()

    board = boards.Board(BOARD_SIZE)
    for idx, node in enumerate(main_sequence, 1):
        move = node.get_move()
        if not move:
            continue
        color, pos = move
        if pos:
            board.play(pos[0], pos[1], color)

    return board, komi, winner


def link_next_move(step_records: list) -> list:
    '''将每一步落子信息和下一步落子信息链接。'''
    board_map = []
    for i in range(len(step_records) - 1):
        cur_step = step_records[i]
        next_step = step_records[i + 1]
        simple_next_step = {
            'color_code': next_step['color_code'],
            'pos': next_step['pos'],
        }
        board_map.append((cur_step, simple_next_step))
    return board_map


def np_input_to_board(np_input: np.ndarray):
    '''将 numpy 矩阵转化为 (board, player) 输出。'''
    board = boards.Board(BOARD_SIZE)
    np_board = np_input[0]
    np_player = np_input[1]

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            val = np_board[r, c]
            if val == 0.0:
                continue
            color = 'b' if val == 1.0 else 'w'
            board.play(r, c, color)

    player = 'b' if np_player[0, 0] == 0.0 else 'w'
    return board, player


def get_sorted_chunk_files(chunk_dir: str) -> list:
    '''获取按索引排序的 chunk 文件路径列表。'''
    chunk_files = []
    for file in os.listdir(chunk_dir):
        if file.startswith('go_dataset') and file.endswith('.npz'):
            chunk_idx = int(file.split('_')[-1].split('.')[0])
            chunk_files.append((chunk_idx, os.path.join(chunk_dir, file)))
    chunk_files.sort(key=lambda x: x[0])
    return [f[1] for f in chunk_files]
