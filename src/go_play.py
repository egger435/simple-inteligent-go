'''
Kata-EgGO 主入口
===============
用法::

    python go_play.py --ai-color w --device cpu
    python go_play.py --config my_config.json
    python go_play.py --save-config config.json          # 导出当前配置模板
    python go_play.py --gui                              # 图形界面模式

配置优先级（低 → 高）：代码默认值 → 配置文件 → 环境变量 → CLI 参数
'''

import sys
import os

# ---- 冻结打包路径处理 ----
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(BASE_DIR)
os.chdir(BASE_DIR)

# ---- Step 1: 解析 CLI 参数（不依赖 ML 模块） ----
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Kata-EgGO: A Simple Intelligent Go Player',
    )

    parser.add_argument('--config', '-cfg',
                        type=str, default=None,
                        help='配置文件路径（JSON 格式，覆盖默认 config.json）')

    parser.add_argument('--save-config', '-sc',
                        type=str, default=None, metavar='PATH',
                        help='将当前配置保存到指定 JSON 文件并退出')

    parser.add_argument('--ai-color', '-c',
                        choices=['b', 'w'],
                        default=None,
                        help='AI 执棋颜色: b - 黑棋; w - 白棋 (默认: w)')

    parser.add_argument('--device', '-d',
                        choices=['cpu', 'cuda'],
                        default=None,
                        help='算法运行设备: cpu / cuda (默认: cuda)')

    parser.add_argument('--have-i', '-hi',
                        action='store_true',
                        help='是否使用包含 I 列的坐标系统')

    parser.add_argument('--top-k', '-tk',
                        type=int, default=None,
                        help='每层搜索保留的 top k 个子节点 (默认: 3)')

    parser.add_argument('--search-depth', '-sd',
                        type=int, default=None,
                        help='MiniMax 搜索深度 (默认: 3)')

    parser.add_argument('--gui', '-g',
                        action='store_true',
                        help='启用图形界面（鼠标点击落子）')

    return parser.parse_args()


args = parse_args()

# ---- Step 2: 导入 common（此时会自动加载默认配置文件和环境变量） ----
import common

# 如果指定了 --config，重新从指定文件加载（覆盖默认配置和环境变量）
if args.config:
    common.config.load_json(args.config)
    # 重新应用环境变量（保持环境变量 > 配置文件 的优先级）
    from config import _apply_env_overrides
    _apply_env_overrides(common.config)
    common._sync_module_vars()

# 应用 CLI 参数覆盖（最高优先级）
common.update_config_from_args(args)

# 如果指定了 --save-config，保存当前配置并退出
if args.save_config:
    common.config.save_json(args.save_config)
    print(f'配置已导出到 {args.save_config}，退出。')
    sys.exit(0)

# ---- Step 3: 现在安全导入模块（配置已生效） ----
from game.player import GoPlayer

if args.gui:
    from ui.app import GoBoardGUI
else:
    from game.loops import vs_ai


# =====================================================================
def main():
    ai_color = args.ai_color or 'w'
    print('== Kata-EgGO: A Simple Intelligent Go Player ==\n')
    print(f'AI 执棋: {ai_color} | 设备: {common.DEVICE}')

    if args.config:
        print(f'配置文件: {args.config}')

    go_player = GoPlayer(ai_player=ai_color)

    if args.gui:
        gui = GoBoardGUI(go_player)
        gui.run()
    else:
        vs_ai(go_player)


if __name__ == '__main__':
    main()
