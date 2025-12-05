import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 导入你的网络定义
# 确保 agents/PolicyNetwork/HexPolicyNet.py 存在
from agents.PolicyNetwork.HexPolicyNet import HexPolicyNet

# --- 配置 ---
DATA_FILE = "data/mcts_games.pt"        # 对应上面的输出文件
MODEL_PATH = "models/hex_policy_v1.pth" # 模型保存路径
BATCH_SIZE = 64
EPOCHS = 15                             # 训练轮数

class HexDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def train():
    # 1. 检查数据
    if not os.path.exists(DATA_FILE):
        print(f"[错误] 找不到数据文件: {DATA_FILE}")
        print("请先运行 python3 generate_mohex_data.py")
        return

    print(f"正在加载数据: {DATA_FILE} ...")
    raw_data = torch.load(DATA_FILE)
    print(f"加载成功! 共 {len(raw_data)} 个样本。")
    
    if len(raw_data) == 0:
        print("数据为空，无法训练。")
        return

    # 2. 准备 DataLoader
    dataset = HexDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    
    # 注意：确保 in_channels 与 Board2Tensor 一致 (默认为 4)
    model = HexPolicyNet(board_size=11, in_channels=4).to(device)
    
    # 尝试加载旧模型(如果存在)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("加载旧模型继续训练...")
        except:
            print("将从头开始训练新模型。")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 4. 训练循环
    print(f"--- 开始训练 ({EPOCHS} Epochs) ---")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        batch_count = 0
        
        for boards, moves in dataloader:
            boards = boards.to(device)
            moves = moves.to(device)
            
            optimizer.zero_grad()
            logits = model(boards)
            loss = criterion(logits, moves)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # 5. 保存模型
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"--- 模型已保存至: {MODEL_PATH} ---")

if __name__ == "__main__":
    train()