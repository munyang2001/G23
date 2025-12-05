import sys
import os
import random
import time
import subprocess

# --- 依赖检查 ---
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("错误: 未检测到 PyTorch 库。请运行: pip install torch")
    sys.exit(1)

from src.Board import Board
from src.Colour import Colour
# 导入队友的 MCTS Agent (你的老师)
from agents.MCTSAgent.MCTSAgent import MCTSAgent
# 导入你的神经网络输入编码器
from agents.PolicyNetwork.Board2Tensor import encode_board_to_tensor

# --- 配置 ---
NUM_GAMES = 200        # 生成多少局数据 (MCTS 较慢，根据时间调整)
OUTPUT_FILE = "data/mcts_games.pt"

# --- 1. 确保二进制文件有权限 ---
def ensure_executable():
    path = "./agents/Group23/NaiveAgent"
    if os.path.exists(path):
        os.chmod(path, 0o755) # 赋予执行权限
        print(f"[系统] 已赋予执行权限: {path}")
    else:
        print(f"[警告] 找不到引擎文件 {path}，请确保已 Pull 队友代码。")

# --- 2. 生成数据 ---
def generate_data():
    ensure_executable()
    
    print(f"--- 开始生成数据: {NUM_GAMES} 局 (MCTS vs MCTS) ---")
    data = []
    
    start_time = time.time()
    
    for i in range(NUM_GAMES):
        try:
            # 初始化 MCTS 引擎
            # 注意：如果 MCTS 内部有状态残留，每局重新初始化更安全
            agent_red = MCTSAgent(Colour.RED)
            agent_blue = MCTSAgent(Colour.BLUE)
        except Exception as e:
            print(f"[错误] MCTS 启动失败: {e}")
            print("请检查是否在 Docker 内运行，且 mcts-hex 路径正确。")
            return

        board = Board(11)
        turn = 1
        curr_colour = Colour.RED
        
        while True:
            # 1. 获取 MCTS 的走法 (Teacher's Move)
            if curr_colour == Colour.RED:
                move = agent_red.make_move(turn, board, None)
            else:
                move = agent_blue.make_move(turn, board, None)
            
            # 2. 记录数据 (忽略 Swap)
            if move.x != -1:
                # 输入: 当前棋盘状态 (Tensor)
                # squeeze(0) 去掉 batch 维度，变成 (C, H, W)
                tensor = encode_board_to_tensor(board, curr_colour).squeeze(0)
                
                # 标签: MCTS 选择的这一步 (Index 0-120)
                target = move.x * 11 + move.y
                
                data.append((tensor, target))
                
                # 执行移动
                board.set_tile_colour(move.x, move.y, curr_colour)
            
            # 3. 检查结束
            # 简单检查：如果棋盘满了或步数过多
            # MCTS Agent 内部通常会处理认输，但这里我们只负责跑流程
            curr_colour = Colour.opposite(curr_colour)
            turn += 1
            if turn > 121: break
        
        # 结束一局，清理进程
        if hasattr(agent_red, 'agent_process'): agent_red.agent_process.terminate()
        if hasattr(agent_blue, 'agent_process'): agent_blue.agent_process.terminate()

        # 打印进度
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            avg = elapsed / (i + 1)
            eta = avg * (NUM_GAMES - i - 1)
            print(f"进度: {i + 1}/{NUM_GAMES} 局 | 样本数: {len(data)} | 耗时: {elapsed:.1f}s | ETA: {eta/60:.1f}min")

    # 保存数据到文件
    if not os.path.exists("data"):
        os.makedirs("data")
    torch.save(data, OUTPUT_FILE)
    print(f"--- 数据生成完毕，已保存至 {OUTPUT_FILE} ---")

if __name__ == "__main__":
    generate_data()