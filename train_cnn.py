"""
Hex 策略网络训练脚本 (升级版)
"""
import sys
import os
import random
import time

# --- 依赖检查 ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("错误: 未检测到 PyTorch 库。请运行: pip install torch")
    sys.exit(1)

from src.Board import Board
from src.Colour import Colour
from agents.Group23.ScoutAgent import ScoutAgent
from agents.PolicyNetwork.HexPolicyNet import HexPolicyNet
from agents.PolicyNetwork.Board2Tensor import encode_board_to_tensor

# --- 配置 ---
NUM_GAMES = 500        # 训练局数：建议至少 100，想变强要 500+ (耗时较长)
BATCH_SIZE = 32
EPOCHS = 10            # 训练轮数
MODEL_PATH = "models/hex_policy_v1.pth"

# --- 1. 数据集生成器 ---
def generate_self_play_data(num_games):
    print(f"--- 开始生成训练数据 ({num_games} 局) ---")
    print("ScoutAgent vs ScoutAgent (自对弈中...)")
    
    data = []
    agent_red = ScoutAgent(Colour.RED)
    agent_blue = ScoutAgent(Colour.BLUE)
    
    start_time = time.time()
    
    for i in range(num_games):
        board = Board(11)
        turn = 1
        curr_colour = Colour.RED
        game_moves = []
        
        while True:
            # 使用 ScoutAgent 的逻辑来决定这一步 (老师)
            if curr_colour == Colour.RED:
                move = agent_red.make_move(turn, board, None)
            else:
                move = agent_blue.make_move(turn, board, None)
            
            # 记录数据 (只记录非 Swap 的正常走子)
            if move.x != -1:
                # 获取棋盘 Tensor (C, H, W)
                tensor = encode_board_to_tensor(board, curr_colour).squeeze(0)
                # 计算移动的 Index (0-120)
                target = move.x * 11 + move.y
                
                # 加入数据集
                data.append((tensor, target))
                
                # 执行移动
                board.set_tile_colour(move.x, move.y, curr_colour)
            
            # 简单判断是否结束 (填满或步数过多)
            curr_colour = Colour.opposite(curr_colour)
            turn += 1
            if turn > 121: 
                break
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"已完成 {i + 1}/{num_games} 局, 累计收集样本: {len(data)}, 耗时: {elapsed:.1f}s")

    print(f"--- 数据生成完毕，共 {len(data)} 个样本 ---")
    return data

# --- 2. Dataset ---
class HexDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# --- 3. 训练主程序 ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型 (4通道输入)
    model = HexPolicyNet(board_size=11, in_channels=4).to(device)
    
    # 如果之前有模型，加载它继续训练 (增量学习)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"加载已有模型继续训练: {MODEL_PATH}")
        except:
            print("未找到旧模型或加载失败，从头开始训练。")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 生成数据
    raw_data = generate_self_play_data(NUM_GAMES)
    if not raw_data: return

    dataset = HexDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"--- 开始训练 (Epochs: {EPOCHS}) ---")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for boards, moves in dataloader:
            boards, moves = boards.to(device), moves.to(device)
            
            optimizer.zero_grad()
            logits = model(boards)
            loss = criterion(logits, moves)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

    # 保存
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存: {MODEL_PATH}")

if __name__ == "__main__":
    train()