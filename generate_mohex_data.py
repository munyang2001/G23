import subprocess
import time
import os
import torch
import numpy as np
from src.Board import Board
from src.Colour import Colour
from agents.PolicyNetwork.Board2Tensor import encode_board_to_tensor

# --- 配置区 ---
# 如果你有编译好的 mohex，填入路径。如果没有，就用你们组的 MCTS。
# 例如: ENGINE_CMD = ["agents/MCTSAgent/mcts-hex"] 
# 或者 MoHex: ENGINE_CMD = ["/path/to/mohex", "--config", "config.txt"]
ENGINE_CMD = ["python3", "Hex.py", "--agent", "MCTSAgent"] # 这是一个占位符，需要替换为真实的启动命令

# 真实场景下，如果 mcts-hex 是编译好的二进制文件：
# ENGINE_CMD = ["./agents/MCTSAgent/mcts-hex"] 

OUTPUT_FILE = "data/mohex_games.pt"
NUM_GAMES = 100
BOARD_SIZE = 11

def send_cmd(proc, command):
    """向 GTP 引擎发送命令并获取回复"""
    proc.stdin.write(f"{command}\n")
    proc.stdin.flush()
    response = ""
    while True:
        line = proc.stdout.readline()
        if line.strip() == "":
            if response: break # 空行表示回复结束
        else:
            response += line
    return response.strip()

def gtp_to_coords(gtp_move):
    """将 GTP 坐标 (e.g., 'C5') 转换为 (x, y)"""
    gtp_move = gtp_move.upper()
    if gtp_move == "SWAP": return (-1, -1)
    if gtp_move == "RESIGN": return None
    
    col_char = gtp_move[0]
    row_str = gtp_move[1:]
    
    # GTP: A=0, B=1, C=2... (跳过 I)
    col = ord(col_char) - ord('A')
    if col_char > 'I': col -= 1
    
    row = int(row_str) - 1
    return (col, row)

def run_self_play():
    data = []
    print(f"启动引擎: {ENGINE_CMD}")
    
    # 启动两个进程
    try:
        p1 = subprocess.Popen(ENGINE_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        p2 = subprocess.Popen(ENGINE_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("错误: 找不到引擎文件。请修改 ENGINE_CMD 路径。")
        return

    # 初始化
    send_cmd(p1, f"boardsize {BOARD_SIZE}")
    send_cmd(p2, f"boardsize {BOARD_SIZE}")

    for i in range(NUM_GAMES):
        print(f"正在进行第 {i+1}/{NUM_GAMES} 局对局...")
        send_cmd(p1, "clear_board")
        send_cmd(p2, "clear_board")
        
        board = Board(BOARD_SIZE)
        curr_player = p1
        curr_colour = Colour.RED
        game_moves = []
        
        while True:
            # 1. 请求当前引擎走棋
            # GTP 命令: genmove red/blue
            color_str = "red" if curr_colour == Colour.RED else "blue"
            response = send_cmd(curr_player, f"genmove {color_str}")
            
            # 解析回复 (通常是 "= C5" 格式)
            move_str = response.split()[-1]
            coords = gtp_to_coords(move_str)
            
            if coords is None: # RESIGN
                break
                
            # 2. 记录数据 (如果是正常走棋)
            if coords != (-1, -1):
                # 记录 (当前局面 Tensor, 这一步 Move)
                tensor = encode_board_to_tensor(board, curr_colour).squeeze(0)
                target = coords[0] * BOARD_SIZE + coords[1]
                data.append((tensor, target))
                
                # 更新 Python 端棋盘
                board.set_tile_colour(coords[0], coords[1], curr_colour)
            
            # 3. 告诉另一个引擎这一步
            other_player = p2 if curr_player == p1 else p1
            send_cmd(other_player, f"play {color_str} {move_str}")
            
            # 4. 切换
            curr_player = other_player
            curr_colour = Colour.opposite(curr_colour)
            
            if len(game_moves) > BOARD_SIZE * BOARD_SIZE: break

    # 保存数据
    if not os.path.exists("data"): os.makedirs("data")
    torch.save(data, OUTPUT_FILE)
    print(f"数据已保存至 {OUTPUT_FILE}，共 {len(data)} 个样本。")
    
    p1.terminate()
    p2.terminate()

if __name__ == "__main__":
    run_self_play()