import torch
import os
import numpy as np
import math

# 基础游戏模块导入
from src.Move import Move
from src.Board import Board
from src.Colour import Colour

# 导入你的父类 Agent (注意路径是 agents.Group23)
from agents.Group23.ScoutAgent import ScoutAgent

# 导入神经网络模块 (注意路径是 agents.PolicyNetwork)
from agents.PolicyNetwork.HexPolicyNet import HexPolicyNet
from agents.PolicyNetwork.Board2Tensor import encode_board_to_tensor

class DeepScout(ScoutAgent):
    """
    DeepScout:
    继承自 ScoutAgent (Alpha-Beta + Dijkstra)。
    区别在于：使用训练好的 CNN 来对 Alpha-Beta 搜索时的候选步进行排序。
    """
    def __init__(self, colour: Colour):
        super().__init__(colour)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 这里的 4 对应目前 Board2Tensor 的输出通道数
        self.model = HexPolicyNet(board_size=11, in_channels=4).to(self.device)
        self.use_cnn = False
        
        # 尝试加载模型
        # 这里的路径是相对于 G23/ 根目录的
        model_path = "models/hex_policy_v1.pth" 
        
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval() # 设为评估模式
                self.use_cnn = True
                # print(f"[DeepScout] CNN 模型加载成功")
            except Exception as e:
                print(f"[DeepScout] 模型加载失败: {e}")
        else:
            print(f"[DeepScout] 未找到模型文件 {model_path}，将退化为普通 ScoutAgent。")

    def _get_ordered_moves(self, board):
        """
        重写：使用 CNN 概率来排序候选步。
        改进策略：混合排序 (Hybrid Ordering)
        """
        # 1. 获取父类的排序结果
        # 父类 ScoutAgent 已经把 "有意义的棋(邻居)" 放在了前面，"无意义的空位" 放在了后面。
        legal_moves = super()._get_ordered_moves(board)
        
        if not self.use_cnn or not legal_moves:
            return legal_moves

        # 2. 准备输入
        try:
            input_tensor = encode_board_to_tensor(board, self.colour).to(self.device)
        except Exception as e:
            return legal_moves
        
        # 3. CNN 预测
        with torch.no_grad():
            probs = self.model.predict_policy(input_tensor)
            probs = probs.cpu().numpy().flatten()

        # 4. 混合筛选策略
        # 如果 CNN 还没训练好，它可能会给这种边缘烂棋很高的分。
        # 为了防止被误导，我们只对父类认为的 "Top 25 候选棋" 进行 CNN 重排。
        # 这样确保了我们永远不会去先搜那些离战场十万八千里的棋子，哪怕 CNN 抽风了。
        
        candidate_count = min(len(legal_moves), 25) 
        candidates = legal_moves[:candidate_count]  # 取出前25步靠谱的棋
        rest = legal_moves[candidate_count:]        # 剩下的垃圾时间棋

        # 只给这 25 步打分排序
        scored_candidates = []
        for move in candidates:
            x, y = move
            idx = x * 11 + y
            score = probs[idx]
            scored_candidates.append((move, score))

        # 降序排列
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 拼接：CNN排序后的好棋 + 剩下的棋
        sorted_moves = [item[0] for item in scored_candidates] + rest
        
        return sorted_moves