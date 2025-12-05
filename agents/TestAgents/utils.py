from src.Board import Board
from src.Move import Move


import random

def make_valid_move(board: Board) -> Move:
    valid_moves = []
    
    # Collect all empty tiles
    for i in range(board.size):
        for j in range(board.size):
            t = board.tiles[i][j]
            if t.colour is None:
                valid_moves.append(Move(i, j))
    
    # If there are valid moves, shuffle and return one
    if valid_moves:
        return random.choice(valid_moves)
    
    return None  # In case there are no valid moves
