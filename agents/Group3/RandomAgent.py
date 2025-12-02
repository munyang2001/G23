from random import choice, random
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Colour import Colour

class RandomAgent(AgentBase):
    """
    一个简单的随机 Agent。
    - 第2回合有 50% 概率 Swap。
    - 其他时候随机选择一个空位下子。
    """
    
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.board_size = 11

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        # 1. Swap 策略 (Turn 2)
        # 50% 概率选择 Swap (-1, -1)
        if turn == 2 and random() > 0.5:
            return Move(-1, -1)
        
        # 2. 获取所有合法移动 (空位)
        available_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                # 检查该位置是否为空 (None)
                if board.tiles[x][y].colour is None:
                    available_moves.append((x, y))
        
        # 3. 随机选择一个
        if not available_moves:
            return Move(-1, -1) # 理论上不会发生，除非棋盘满了
            
        x, y = choice(available_moves)
        return Move(x, y)