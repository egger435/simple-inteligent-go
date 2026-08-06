'''共享 KataGo 引擎模块
提供 KataGo 子进程管理和局面价值查询功能。
'''

import atexit
import queue
import subprocess
import sys
import threading
import time
import json
from config import config


class KataGoEngine:
    '''KataGo 分析引擎交互类。

    管理 katago.exe 子进程的生命周期，通过 JSON 协议发送局面查询，
    返回指定行棋方的胜率。

    通常不直接实例化，改用 get_katago_engine() 获取全局单例::

        engine = get_katago_engine()
        winrate = engine.get_value('b', [['B', 'Q16'], ['W', 'D4']])
    '''

    def __init__(self):
        self.process = None
        self.request_id = 0
        self._line_queue = queue.Queue()   # stdout 行缓冲队列
        self._reader_thread = None
        self._start_engine()
        self._start_reader()

    # ------------------------------------------------------------------
    def _start_engine(self) -> None:
        '''启动 KataGo 子进程并等待就绪信号。'''
        try:
            self.process = subprocess.Popen(
                [
                    config.kata_exe_path,
                    'analysis',
                    '-model', config.kata_model_path,
                    '-config', config.kata_config_path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

            ready = False
            # 此时后台 reader 尚未启动，直接读 stdout
            for _ in range(300):
                line = self.process.stdout.readline()
                if not line:
                    continue
                if 'ready to begin handling requests' in line:
                    ready = True
                    break

            if not ready:
                raise RuntimeError('KataGo 启动超时')

        except Exception as e:
            print(f'KataGo 启动失败: {e}')
            sys.exit(1)

    # ------------------------------------------------------------------
    def restart(self) -> None:
        '''关闭并重新启动 KataGo 引擎。'''
        self.close()
        self._start_engine()
        print('KataGo 已重启')

    # ------------------------------------------------------------------
    def close(self) -> None:
        '''终止 KataGo 子进程。'''
        if self.process:
            print('关闭 KataGo...')
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    # ------------------------------------------------------------------
    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    def _start_reader(self) -> None:
        '''启动后台线程持续读取 KataGo stdout 到队列。'''
        def _reader():
            try:
                for line in self.process.stdout:
                    self._line_queue.put(line)
            except Exception as e:
                print(f'[KataGo] 读取线程异常退出: {e}')
        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

    # ------------------------------------------------------------------
    def _read_line(self, timeout: float) -> str | None:
        '''从队列读取一行（超时返回 None）。'''
        try:
            return self._line_queue.get(timeout=timeout)
        except queue.Empty:
            if not self._reader_thread.is_alive():
                print('[KataGo] 读取线程已停止！')
            return None

    # ------------------------------------------------------------------
    def get_value(self, root_player: str, moves: list) -> float:
        '''根据当前落子序列查询 root_player 视角的胜率。

        Args:
            root_player: 根节点行棋方 ('b' 或 'w')
            moves: 落子序列，格式 [['B', 'Q16'], ['W', 'D4'], ...]

        Returns:
            float: root_player 视角的胜率 (0.0 ~ 1.0)；
                   网络错误或超时返回 -1；KataGo 错误返回 -2
        '''
        self.request_id += 1
        request = {
            'id': str(self.request_id),
            'boardXSize': config.board_size,
            'boardYSize': config.board_size,
            'initialStones': [],
            'moves': moves,
            'rules': 'chinese',
            'komi': 7.5,
            'visits': 10,
            'includePolicy': False,
            'includeOwnership': False,
            'includeMovesOwnership': False,
        }

        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()

        response = None
        start_read = time.time()
        while time.time() - start_read < 5:
            line = self._read_line(0.5)
            if not line:
                continue

            kata_message = line.strip()
            print(f'[KataGo] {kata_message}')

            # 错误检测：含 error 关键字，但不含误报的 rawStWrError
            if ('error' in kata_message.lower()) and 'rawStWrError' not in kata_message:
                print(f'KataGo 返回错误: {kata_message}')
                return -2  # 不重启，返回错误码

            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                # 有些响应带前缀 "800: Error: ..." → 尝试提取 JSON 部分
                if ':' in kata_message:
                    try:
                        json_start = kata_message.index('{')
                        resp = json.loads(kata_message[json_start:])
                    except (ValueError, json.JSONDecodeError):
                        continue
                else:
                    continue

            if 'error' in resp:
                print(f"KataGo 返回错误: {resp['error']}")
                return -2
            if 'rootInfo' not in resp:
                continue
            response = resp
            break

        if not response:
            print('KataGo 响应超时')
            return -1

        # 解析结果
        next_player = 'b' if len(moves) % 2 == 0 else 'w'
        winrate = response['rootInfo']['winrate']
        print(f'我方胜率: {100 * (1 - winrate):.2f}%\n')

        return winrate if root_player == next_player else (1 - winrate)

    # ------------------------------------------------------------------
    def get_final_score(self, moves: list) -> float:
        '''查询终局目数差（正 = 黑好, 负 = 白好），用于自对弈胜负判定。

        Args:
            moves: 落子序列，格式 [['B', 'Q16'], ['W', 'D4'], ...]

        Returns:
            float: 黑方领先的目数（含贴目）。正值为黑胜，负值为白胜。
                   错误返回 None
        '''
        self.request_id += 1
        request = {
            'id': str(self.request_id),
            'boardXSize': config.board_size,
            'boardYSize': config.board_size,
            'initialStones': [],
            'moves': moves,
            'rules': 'chinese',
            'komi': 7.5,
            'visits': 1,           # 终局只需 1 次 visit，大幅加速
            'includePolicy': False,
            'includeOwnership': False,
            'includeMovesOwnership': False,
        }

        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()

        # 检查进程是否存活
        if self.process.poll() is not None:
            print(f'[KataGo] 进程已退出，exitcode={self.process.returncode}')
            return None

        response = None
        start_read = time.time()
        while time.time() - start_read < 10:
            line = self._read_line(0.5)
            if not line:
                continue

            try:
                resp = json.loads(line)
                if 'error' in resp:
                    print(f"KataGo 错误: {resp['error']}")
                    return None
                if 'rootInfo' not in resp:
                    continue
                response = resp
                break
            except json.JSONDecodeError:
                continue

        if not response:
            alive = self._reader_thread.is_alive()
            proc_alive = self.process.poll() is None
            qsize = self._line_queue.qsize()
            print(f'[KataGo] 终局查询超时 | '
                  f'进程存活={proc_alive} 读取线程存活={alive} '
                  f'队列积压={qsize} 步数={len(moves)}')
            return None

        # scoreLead: 当前行棋方领先的目数
        # 终局时 moves 长度决定了 "当前行棋方" = 下一步该走的人
        next_player = 'b' if len(moves) % 2 == 0 else 'w'
        score_lead = response['rootInfo'].get('scoreLead', 0)

        # 转为黑方视角的目数差
        return score_lead if next_player == 'b' else -score_lead



    # ------------------------------------------------------------------
    def get_value_batch(self, queries: list) -> list:
        """流水线批量查询。queries: [(root_player, moves), ...]"""
        # 发送所有请求
        for root_player, moves in queries:
            self.request_id += 1
            request = {
                'id': str(self.request_id), 'boardXSize': config.board_size,
                'boardYSize': config.board_size, 'initialStones': [],
                'moves': moves, 'rules': 'chinese', 'komi': 7.5,
                'visits': 10, 'includePolicy': False,
                'includeOwnership': False, 'includeMovesOwnership': False,
            }
            self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()

        # 收集响应
        results = [None] * len(queries)
        received = 0
        t0 = time.time()
        while received < len(queries) and time.time() - t0 < 15:
            line = self._read_line(0.5)
            if not line:
                continue
            msg = line.strip()
            if ('error' in msg.lower()) and 'rawStWrError' not in msg:
                results[received] = -2; received += 1; continue
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                if ':' in msg:
                    try:
                        resp = json.loads(msg[msg.index('{'):])
                    except: continue
                else: continue
            if 'error' in resp:
                results[received] = -2; received += 1; continue
            if 'rootInfo' not in resp: continue
            rp, moves = queries[received]
            np = 'b' if len(moves) % 2 == 0 else 'w'
            wr = resp['rootInfo']['winrate']
            results[received] = wr if rp == np else (1 - wr)
            received += 1

        for i, r in enumerate(results):
            if r is None: results[i] = -1
        return results

# =====================================================================
# 全局单例（整个对局过程中复用同一个 KataGo 进程）
# =====================================================================

_katago_engine: KataGoEngine | None = None


def get_katago_engine() -> KataGoEngine:
    '''获取全局 KataGoEngine 单例（首次调用时启动子进程）。

    后续每次 AI 走棋时复用同一个进程，避免每次走棋都启动新的 katago.exe。
    '''
    global _katago_engine
    if _katago_engine is None:
        _katago_engine = KataGoEngine()
    return _katago_engine


def shutdown_katago_engine() -> None:
    '''关闭全局 KataGo 引擎（程序退出时自动调用）。'''
    global _katago_engine
    if _katago_engine is not None:
        _katago_engine.close()
        _katago_engine = None


# 注册退出时自动清理
atexit.register(shutdown_katago_engine)
