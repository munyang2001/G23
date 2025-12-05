import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 导入你的 CNN 模型定义
from agents.PolicyNetwork.HexPolicyNet import HexPolicyNet

# --- 配置 ---
DATA_FILE = "data/mcts_games.pt"       # 数据文件路径
MODEL_PATH = "models/hex_policy_v1.pth" # 模型保存路径
BATCH_SIZE = 64
EPOCHS = 20                            # 训练轮数

# --- Dataset 定义 ---
class HexDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def train():
    # 1. 检查数据文件
    if not os.path.exists(DATA_FILE):
        print(f"[错误] 未找到数据文件: {DATA_FILE}")
        print("请先运行 'python3 generate_mohex_data.py' 生成数据。")
        return

    print(f"正在加载数据: {DATA_FILE} ...")
    raw_data = torch.load(DATA_FILE)
    print(f"加载成功，共 {len(raw_data)} 个样本。")

    dataset = HexDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    # 注意: in_channels 必须与 Board2Tensor 的输出一致 (目前是 4)
    model = HexPolicyNet(board_size=11, in_channels=4).to(device)
    
    # 尝试加载旧模型 (断点续传)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("加载旧模型继续训练...")
        except:
            print("未找到旧模型，从头开始训练。")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 3. 开始训练
    print(f"--- 开始训练 ({EPOCHS} 轮) ---")
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
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # 4. 保存模型
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"--- 模型已保存至: {MODEL_PATH} ---")
    print("现在你可以使用 DeepScout (你的 Agent) 加载这个模型进行对战了。")

if __name__ == "__main__":
    train()