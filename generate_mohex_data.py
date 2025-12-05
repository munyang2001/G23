import sys
import os
import time
import torch
import numpy as np

# 依赖检查
try:
    from torch.utils.data import Dataset
except ImportError:
    print("错误: 请安装 PyTorch (pip install torch)")
    sys.exit(1)

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# 导入你 Group23 的 MCTS Agent (文件名叫 NaiveAgent.py, 类名叫 Agent)
from agents.Group23.NaiveAgent import Agent as Group23Agent
from agents.Group23.board_set import Board_Optimized

# 导入 Tensor 转换器
from agents.PolicyNetwork.Board2Tensor import encode_board_to_tensor

# --- 配置 ---
NUM_GAMES = 50         # 生成局数
MOVE_TIME_LIMIT = 0.5   # 每步思考时间 (秒)。原版是8.5s，这里为了生成速度设为0.5s
OUTPUT_FILE = "data/mcts_games.pt"

# --- 快速版 Agent (继承自你们的代码) ---
class FastDataAgent(Group23Agent):
    """
    继承你们的 MCTS Agent，但允许修改思考时间，以便快速生成数据。
    """
    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        # 1. 处理 Swap 逻辑
        
        if turn == 2:
            opp_r, opp_c = -1, -1
            for r in range(self._board_size):
                for c in range(self._board_size):
                    if board.tiles[r][c].colour is not None:
                        opp_r, opp_c = r, c
            # 如果对手下在中心区域 (2,2) 到 (8,8)，则交换
            if 2 <= opp_r <= 8 and 2 <= opp_c <= 8:
                return Move(-1, -1)

        # 2. 更新 MCTS 树根 (对手的走法)
        if opp_move is not None:
            opp_move_tuple = (opp_move.x, opp_move.y) # 注意：这里用 x, y 属性
            self.mcts.update_root(opp_move_tuple)

        # 3. 转换棋盘
        optimized_board = Board_Optimized.from_game_board(board, self.colour)

        # 4. 检查是否只有唯一合法步 (优化)
        legal_moves = list(optimized_board.get_legal_moves())
        if len(legal_moves) == 1:
            return Move(legal_moves[0][0], legal_moves[0][1])

        # 5. 执行 MCTS 搜索 (使用自定义的快速时间限制)
        row, col = self.mcts.search(optimized_board, self.colour, time_limit=MOVE_TIME_LIMIT)
        
        # 6. 更新 MCTS 树根 (自己的走法)
        my_move_tuple = (row, col)
        self.mcts.update_root(my_move_tuple)
        
        return Move(row, col)

# --- 生成数据主循环 ---
def generate_data():
    print(f"--- 开始生成数据: {NUM_GAMES} 局 (Fast MCTS vs Fast MCTS) ---")
    print(f"--- 每步思考时间: {MOVE_TIME_LIMIT} 秒 ---")
    
    data = []
    start_time = time.time()
    
    for i in range(NUM_GAMES):
        # 初始化两个快速 MCTS Agent
        agent_red = FastDataAgent(Colour.RED)
        agent_blue = FastDataAgent(Colour.BLUE)
        
        board = Board(11)
        turn = 1
        curr_colour = Colour.RED
        last_move = None
        
        while True:
            # 获取当前玩家的 Agent
            current_agent = agent_red if curr_colour == Colour.RED else agent_blue
            
            # 让 Agent 思考
            move = current_agent.make_move(turn, board, last_move)
            
            # 记录数据 (跳过 Swap)
            if move.x != -1:
                # 输入: 当前盘面 Tensor
                # squeeze(0) 是为了去掉 batch 维度，保存为 (C, H, W)
                tensor = encode_board_to_tensor(board, curr_colour).squeeze(0)
                
                # 标签: MCTS 算出来的最佳走法 (Index 0-120)
                target = move.x * 11 + move.y
                
                data.append((tensor, target))
                
                # 执行移动
                board.set_tile_colour(move.x, move.y, curr_colour)
            else:
                # 处理 Swap: 交换颜色，但不记录数据
                # 简单起见，我们在生成数据时可以忽略 Swap 的逻辑处理，
                # 或者简单地视为当前玩家继续执红/蓝
                pass

            # 检查胜负
            if board.has_ended(curr_colour):
                break
            
            # 切换回合
            last_move = move
            curr_colour = Colour.opposite(curr_colour)
            turn += 1
            if turn > 121: break
        
        # 打印进度
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            avg_per_game = elapsed / (i + 1)
            eta = avg_per_game * (NUM_GAMES - i - 1)
            print(f"进度: {i + 1}/{NUM_GAMES} 局 | 样本数: {len(data)} | 耗时: {elapsed:.1f}s | ETA: {eta/60:.1f}min")

    # 保存文件
    if not os.path.exists("data"):
        os.makedirs("data")
    
    torch.save(data, OUTPUT_FILE)
    print(f"--- 数据生成完毕! 已保存至 {OUTPUT_FILE} ---")
    print(f"--- 总样本数: {len(data)} ---")

if __name__ == "__main__":
    generate_data()