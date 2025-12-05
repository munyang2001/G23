import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from agents.PolicyNetwork.HexPolicyNet import HexPolicyNet

# --- 配置 ---
DATA_FILE = "data/mohex_games.pt"
MODEL_PATH = "models/hex_policy_v1.pth"
BATCH_SIZE = 64
EPOCHS = 20

class HexDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def train():
    if not os.path.exists(DATA_FILE):
        print(f"错误: 未找到数据文件 {DATA_FILE}")
        print("请先运行 generate_mohex_data.py 生成数据。")
        return

    print(f"加载数据: {DATA_FILE} ...")
    raw_data = torch.load(DATA_FILE)
    print(f"加载完成，共 {len(raw_data)} 个样本。")

    dataset = HexDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexPolicyNet(board_size=11, in_channels=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    print("开始训练...")
    
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

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型保存至: {MODEL_PATH}")

if __name__ == "__main__":
    train()