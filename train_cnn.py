import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 确保能导入 agents
sys.path.append(os.getcwd())
from agents.PolicyNetwork.HexPolicyNet import HexPolicyNet

# --- 配置 ---
DATA_FILE = "data/mcts_advanced_games.pt"
MODEL_PATH = "models/hex_policy_v2.pth"
BATCH_SIZE = 64
EPOCHS = 20

class HexDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # data格式: (board, policy_target, value_target)
        return self.data[idx]

def train():
    if not os.path.exists(DATA_FILE):
        print(f"错误: 找不到 {DATA_FILE}。请先在 Docker 里运行 generate_mohex_data.py")
        return

    print("正在加载数据...")
    raw_data = torch.load(DATA_FILE)
    print(f"加载成功! 样本数: {len(raw_data)}")
    
    dataset = HexDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # 初始化模型
    model = HexPolicyNet(board_size=11, in_channels=4).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 两个 Loss：
    # 1. Policy Loss (Cross Entropy with Soft Targets) -> 预测走哪一步
    # 2. Value Loss (MSE) -> 预测输赢 (如果你的网络还没有 value head，可以暂时忽略这个，只训 Policy)
    
    print("开始训练...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for boards, policies, values in dataloader:
            boards = boards.to(device)
            target_policies = policies.to(device) # 概率分布
            # target_values = values.to(device) # 暂时没用到，等网络升级 Value Head
            
            optimizer.zero_grad()
            
            # Forward
            logits = model(boards) # (Batch, 121)
            
            # Loss: KL Divergence or CrossEntropy
            # 因为 target 是概率分布，我们用 LogSoftmax + KLDiv
            log_probs = F.log_softmax(logits, dim=1)
            
            # KLDivLoss expectation: input=log_probs, target=probs
            loss = F.kl_div(log_probs, target_policies, reduction='batchmean')
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Policy Loss: {total_loss/len(dataloader):.4f}")

    if not os.path.exists("models"): os.makedirs("models")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存至 {MODEL_PATH}")

if __name__ == "__main__":
    train()