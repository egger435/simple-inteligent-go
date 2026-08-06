'''
中心配置模块
使用 dataclass 管理所有配置项, 支持 JSON 配置文件、环境变量覆盖和命令行参数更新。

优先级（低→高）：代码默认值 → config.json → 环境变量 → CLI 参数
'''

import json
import os
from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class Config:
    '''围棋 AI 全局配置'''

    # ===== 核心设置 =====
    device: str = 'cuda'
    '''运行设备 (cuda / cpu)'''
    board_size: int = 19
    '''棋盘大小'''
    pass_label: int = 361
    '''弃行标签'''
    have_i: bool = False
    '''棋盘坐标是否包含 I 列'''
    color_map: dict = field(default_factory=lambda: {1: 0.0, 2: 1.0})
    '''行棋方数值映射'''

    # ===== 路径设置 =====
    sgf_dataset_dir: str = r'data\gogod_commentary_sgfs'
    '''原始 SGF 数据目录'''

    # 策略网络数据集
    sg_dataset_save_path: str = r'E:\go_dataset\strategy_net'
    '''策略网络数据集保存位置'''
    sg_chunk_sample_num: int = 500000
    '''每 chunk 样本数'''

    # 价值网络数据集
    va_dataset_save_path: str = r'E:\go_dataset\value_net_dataset_n.npz'
    '''价值网络数据集保存位置'''

    # 日志与模型输出
    log_path: str = r'output_models\log_resnet.txt'
    '''训练日志保存位置'''

    # ===== 策略网络模型 =====
    sg_chunk_dir: str = r'E:\go_dataset\strategy_net'
    '''策略网络分片数据集目录'''
    sg_valid_chunk_dir: str = r'E:\go_dataset\strategy_net\valid_chunk.npz'
    '''策略网络验证集路径'''
    sg_save_model_path: str = r'output_models\go_strategy_model_1_2.pth'
    '''策略网络模型保存路径'''
    sg_best_model_path: str = r'output_models\go_strategy_model_t_b.pth'
    '''策略网络最佳模型保存路径'''
    sg_valid_interval: int = 1
    '''每训练几个 chunk 验证一次'''
    sg_scheduler_patience: int = 5
    '''学习率调度容忍次数'''
    sg_scheduler_factor: float = 0.5
    '''学习率衰减系数'''
    sg_num_global_epochs: int = 3
    '''全局 epoch 数'''
    sg_batch_size: int = 32
    '''策略网络训练 batch size'''
    sg_epochs_per_chunk: int = 3
    '''每个 chunk 训练 epoch 数'''
    sg_learning_rate: float = 1e-4
    '''策略网络学习率'''
    sg_weight_decay: float = 1e-5
    '''策略网络权重衰减'''
    sg_gradient_accumulation_steps: int = 2
    '''梯度累积步数'''
    sg_gradient_clip: float = 1.0
    '''梯度裁剪阈值'''

    # ===== 价值网络模型 =====
    va_dataset_path: str = r'E:\go_dataset\value_net_dataset.npz'
    '''价值网络数据集路径'''
    va_save_model_path: str = r'output_models\go_final_val_model_1_2.pth'
    '''价值网络模型保存路径'''
    va_batch_size: int = 128
    '''价值网络训练 batch size'''
    va_learning_rate: float = 1e-3
    '''价值网络学习率'''
    va_weight_decay: float = 1e-4
    '''价值网络权重衰减'''
    va_patience: int = 35
    '''价值网络早停容忍次数'''

    # ===== 策略选择（推理）=====
    strategy_model_path: str = r'output_models\go_strategy_model_1_1.pth'
    '''策略网络模型权重路径'''
    strategy_model_type: str = 'GoCNN_p'
    '''策略网络模型类型 (GoCNN | GoCNN_p | GoCNN_t | AlphaCNN)'''
    top_k: int = 3
    '''最优候选落子个数'''

    # ===== 价值判断（推理）=====
    value_model_path: str = r'output_models\go_final_val_model_1_2.pth'
    '''价值网络模型权重路径'''

    # ===== KataGo 引擎 =====
    kata_exe_path: str = r'katago\katago.exe'
    '''KataGo 可执行文件路径'''
    kata_model_path: str = r'katago\kata1-b40c256-s5109387264-d1232289301.bin.gz'
    '''KataGo 权重文件路径'''
    kata_config_path: str = r'katago\analysis_config.cfg'
    '''KataGo 配置文件路径'''

    # ===== 搜索算法 =====
    max_search_depth: int = 3
    '''MiniMax 最大搜索深度'''
    mc_start_threshold: int = 150
    '''蒙特卡洛推演启动阈值（步数）'''
    mc_simulations: int = 10
    '''蒙特卡洛推演次数'''
    use_own_value_net: bool = False
    '''使用自训练价值网络替代 KataGo 做局面评估'''
    mc_rollouts: int = 5
    '''每叶节点 MC 推演次数'''
    mc_steps: int = 5
    '''每次 MC 推演步数'''

    # ===== MCTS 搜索 =====
    search_algorithm: str = 'minimax'
    '''搜索算法: minimax 或 mcts'''
    mcts_simulations: int = 100
    '''每次走棋 MCTS 模拟次数'''
    mcts_c_puct: float = 1.4
    '''MCTS UCB 常数'''
    mcts_expand_width: int = 20
    '''MCTS 每节点扩展的最大分支数'''
    mcts_eval_batch_size: int = 10
    '''MCTS 叶节点批量评估批次大小（KataGo 模式加速）'''
    mcts_temperature: float = 0.0
    '''MCTS 落子温度 (0=贪心选访问最多)'''
    mcts_visualize: bool = False
    '''GUI 模式是否可视化 MCTS 搜索过程'''
    mcts_vis_interval: float = 0.5
    '''MCTS 可视化每步模拟间隔（秒）'''
    mcts_verbose: bool = False
    '''是否打印 MCTS 搜索过程（扩展/评估/路径）'''

    # ===== 自对弈数据标注 =====
    sp_games: int = 500
    '''自对弈标注局数'''
    sp_sample_interval: int = 5
    '''每 N 手采一个样本'''
    sp_temperature: float = 0.8
    '''自对弈温度'''
    sp_katago_visits: int = 50
    '''KataGo 标注时的 visits'''
    sp_output_path: str = r'E:\go_dataset\selfplay_value_dataset.npz'
    '''自对弈标注数据集输出路径'''

    # ===== 强化学习训练 =====
    rl_games_per_generation: int = 100
    '''每代自对弈局数'''
    rl_buffer_capacity: int = 500000
    '''经验回放缓冲区容量'''
    rl_batch_size: int = 256
    '''RL 训练 batch size'''
    rl_learning_rate: float = 1e-5
    '''RL 学习率'''
    rl_train_steps_per_gen: int = 500
    '''每代训练步数'''
    rl_temperature_early: float = 1.0
    '''自对弈前期温度（高=探索）'''
    rl_temperature_late: float = 0.5
    '''自对弈后期温度（低=利用）'''
    rl_eval_games: int = 100
    '''Arena 评估对战局数'''
    rl_accept_threshold: float = 0.55
    '''新模型接受阈值（胜率超过此值才替换）'''
    rl_max_generations: int = 20
    '''最大训练代数'''

    # ===== 内部状态（不导出）=====
    _strategy_model: object = field(default=None, repr=False, init=False)
    _value_model: object = field(default=None, repr=False, init=False)

    # 以下字段仅用于训练，不出现在 config.json 中
    _json_exclude = {
        'sgf_dataset_dir',
        'sg_dataset_save_path', 'sg_chunk_sample_num',
        'va_dataset_save_path', 'log_path',
        'sg_chunk_dir', 'sg_valid_chunk_dir',
        'sg_save_model_path', 'sg_best_model_path',
        'sg_valid_interval', 'sg_scheduler_patience', 'sg_scheduler_factor',
        'sg_num_global_epochs', 'sg_batch_size', 'sg_epochs_per_chunk',
        'sg_learning_rate', 'sg_weight_decay',
        'sg_gradient_accumulation_steps', 'sg_gradient_clip',
        'va_dataset_path', 'va_save_model_path',
        'va_batch_size', 'va_learning_rate', 'va_weight_decay', 'va_patience',
        # 自对弈数据标注（训练脚本专用）
        'sp_games', 'sp_sample_interval', 'sp_temperature',
        'sp_katago_visits', 'sp_output_path',
        # 强化学习（训练脚本专用）
        'rl_games_per_generation', 'rl_buffer_capacity', 'rl_batch_size',
        'rl_learning_rate', 'rl_train_steps_per_gen',
        'rl_temperature_early', 'rl_temperature_late',
        'rl_eval_games', 'rl_accept_threshold', 'rl_max_generations',
        # 旧版 MinimaxMCR 专用（当前 BatchMinimaxMCR 不使用）
        'mc_start_threshold', 'mc_simulations',
    }

    # ------------------------------------------------------------------
    def update_from_args(self, args) -> None:
        '''从 argparse 解析结果更新配置。
        必须在任何模型加载之前调用。'''
        # 普通参数：值非 None 时覆盖
        for attr in ('device', 'top_k', 'max_search_depth', 'ai_color'):
            if hasattr(args, attr):
                val = getattr(args, attr)
                if val is not None:
                    setattr(self, attr, val)

        # store_true 布尔参数：只有显式传入 True 时才覆盖
        # （argparse 默认值是 False，不能因为用户没传就覆盖配置文件）
        for attr in ('have_i',):
            if hasattr(args, attr) and getattr(args, attr) is True:
                setattr(self, attr, True)

    # ------------------------------------------------------------------
    def load_json(self, path: str) -> None:
        '''从 JSON 配置文件加载配置，覆盖已有属性。

        跳过以 ``_`` 开头的内部字段和 JSON 中有但 Config 中不存在的键。
        不存在的文件静默跳过。
        自动转换 dict 字段的字符串键为原始键类型（JSON 不支持整数键）。
        '''
        if not os.path.exists(path):
            return

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for key, value in data.items():
            if key.startswith('_'):
                continue
            if not hasattr(self, key):
                print(f'[Config] 忽略未知配置项: {key}')
                continue
            value = self._coerce_json_value(key, value)
            setattr(self, key, value)

        print(f'[Config] 已从 {path} 加载配置')

    # ------------------------------------------------------------------
    def _coerce_json_value(self, key: str, value):
        '''将 JSON 值转换为与目标字段兼容的类型。

        主要处理 JSON 无法表达的 Python 类型，例如：
        - dict 字段的整数键在 JSON 中变为字符串，需要还原。
        '''
        current = getattr(self, key)

        # dict 键类型还原：JSON 的字符串键 → 原始键类型（如 int）
        if isinstance(current, dict) and isinstance(value, dict) and current:
            key_type = type(next(iter(current.keys())))
            if key_type is not str:
                converted = {}
                for k, v in value.items():
                    try:
                        converted[key_type(k)] = v
                    except (ValueError, TypeError):
                        converted[k] = v  # 还原失败则保留原始字符串键
                return converted

        return value

    # ------------------------------------------------------------------
    def save_json(self, path: str) -> None:
        '''将当前配置保存为 JSON 文件（跳过内部字段和训练专用字段）。'''
        data = {}
        for f in fields(self):
            if f.name.startswith('_'):
                continue
            if not f.init:
                continue
            if f.name in self._json_exclude:
                continue
            data[f.name] = getattr(self, f.name)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f'[Config] 配置已保存到 {path}')

    def create_strategy_model(self):
        '''懒加载策略网络模型，使用当前 device 和 model_path 设置。'''
        import torch
        import models.strategy_net as sg

        model_classes = {
            'GoCNN':    sg.GoCNN,
            'GoCNN_p':  sg.GoCNN_p,
            'GoCNN_t':  sg.GoCNN_t,
            'AlphaCNN': sg.AlphaCNN,
        }
        model_cls = model_classes.get(self.strategy_model_type, sg.GoCNN_p)
        model = model_cls().to(self.device)

        checkpoint = torch.load(self.strategy_model_path, map_location=self.device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model

    def create_value_model(self):
        '''懒加载价值网络模型，使用当前 device 和 value_model_path 设置。'''
        import torch
        import models.value_net as va

        model = va.GoValueNet().to(self.device)
        state_dict = torch.load(self.value_model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model


# ===== 全局配置实例 =====
config = Config()


# ===== 配置文件加载 =====
def _load_config_file(cfg: Config) -> None:
    '''从 JSON 配置文件加载配置。

    路径优先级：GO_AI_CONFIG 环境变量 > 项目根目录 config.json。
    文件不存在则静默跳过（向后兼容无配置文件运行）。
    '''
    config_path = os.environ.get('GO_AI_CONFIG', 'config.json')
    if os.path.exists(config_path):
        cfg.load_json(config_path)


_load_config_file(config)


# ===== 环境变量覆盖 =====
def _apply_env_overrides(cfg: Config) -> None:
    '''从环境变量覆盖配置（在模块加载时调用一次）。'''
    env_map = {
        'GO_AI_DEVICE':               'device',
        'GO_AI_TOP_K':                'top_k',
        'GO_AI_SEARCH_DEPTH':         'max_search_depth',
        'GO_AI_STRATEGY_MODEL_PATH':  'strategy_model_path',
        'GO_AI_VALUE_MODEL_PATH':     'value_model_path',
        'GO_AI_KATA_EXE_PATH':        'kata_exe_path',
        'GO_AI_KATA_MODEL_PATH':      'kata_model_path',
        'GO_AI_KATA_CONFIG_PATH':     'kata_config_path',
        'GO_AI_SG_DATASET_DIR':       'sg_dataset_save_path',
        'GO_AI_VA_DATASET_PATH':      'va_dataset_path',
        'GO_AI_LOG_PATH':             'log_path',
    }
    for env_var, attr in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            # 类型转换
            current = getattr(cfg, attr)
            if isinstance(current, bool):
                val = val.lower() in ('1', 'true', 'yes')
            elif isinstance(current, int):
                val = int(val)
            elif isinstance(current, float):
                val = float(val)
            setattr(cfg, attr, val)


_apply_env_overrides(config)
