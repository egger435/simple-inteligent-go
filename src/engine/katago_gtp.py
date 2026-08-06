'''KataGo GTP 客户端 —— 作为对局对手（支持 -level 控制棋力）。

GTP 协议是文本问答：发 "genmove b" → 收 "= D4"。用作 ELO 评测的基准对手。

用法::

    opp = KataGoGTPClient(level=4)
    move = opp.genmove('b')        # 对手走棋 → 'D4' 或 'pass'
    opp.play('w', 'D4')            # 通知对手我方落子
    score = opp.final_score()      # 终局判定 → 'B+3.5'
'''

import queue
import subprocess
import threading
import time
from config import config


class KataGoGTPClient:
    '''KataGo GTP 协议客户端。

    用 maxVisits 控制基准棋力（visits 越多越强），
    配合 -override-config 覆盖 config 中的 defaultVisits。
    '''

    def __init__(self, visits: int = 10, timeout: float = 60):
        self.visits = visits
        self.timeout = timeout
        self._line_queue = queue.Queue()
        self._start_process()
        # 初始化对局设置
        self._send('boardsize 19')
        self._send('komi 7.5')
        self._send('clear_board')

    # ------------------------------------------------------------------
    def _start_process(self):
        self.proc = subprocess.Popen(
            [
                config.kata_exe_path,
                'gtp',
                '-model', config.kata_model_path,
                '-override-config',
                f'maxVisits={self.visits},ponderingEnabled=false',
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        threading.Thread(target=self._reader, daemon=True).start()

    # ------------------------------------------------------------------
    def _reader(self):
        try:
            for line in self.proc.stdout:
                self._line_queue.put(line)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _send(self, cmd: str, timeout: float = None) -> list:
        '''发送 GTP 命令，读取响应直到空行。'''
        if timeout is None:
            timeout = self.timeout
        self.proc.stdin.write(cmd + '\n')
        self.proc.stdin.flush()

        lines = []
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                line = self._line_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            s = line.strip()
            if s == '':
                break
            lines.append(s)
        return lines

    # ------------------------------------------------------------------
    def genmove(self, color: str) -> str:
        '''让对手走棋，返回 'D4' 或 'pass'。'''
        lines = self._send(f'genmove {color}')
        if not lines:
            return 'pass'
        first = lines[0]
        if first.startswith('='):
            return first[2:].strip() or 'pass'
        return 'pass'

    # ------------------------------------------------------------------
    def clear_board(self):
        '''重置棋盘（新对局开始前必须调用）。'''
        self._send('clear_board')

    # ------------------------------------------------------------------
    def play(self, color: str, pos: str):
        '''通知对手我方落子。pos 如 'D4' 或 'pass'。'''
        self._send(f'play {color} {pos}')

    # ------------------------------------------------------------------
    def final_score(self) -> str:
        '''终局判定，返回 'B+3.5' / 'W+2' / '0'。'''
        lines = self._send('final_score')
        if lines and lines[0].startswith('='):
            return lines[0][2:].strip()
        return '0'

    # ------------------------------------------------------------------
    def close(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
