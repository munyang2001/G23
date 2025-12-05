import random
import copy
import numpy as np
from src.Colour import Colour
from src.Tile import Tile

class ZobristHash:
    def __init__(self, board_size=11):
        self.board_size = board_size
        self.hash = 0
        self.turn = random.getrandbits(64)
        self.table = {}
        for r in range(board_size):
            for c in range(board_size):
                self.table[(r, c, Colour.RED)] = random.getrandbits(64)
                self.table[(r, c, Colour.BLUE)] = random.getrandbits(64)

    def update_cell(self, row, col, colour: Colour):
        if colour in [Colour.RED, Colour.BLUE]:
            self.hash ^= self.table[(row, col, colour)]

    def update_turn(self):
        self.hash ^= self.turn

    def get_hash(self):
        return self.hash

class Board_Optimized:
    EMPTY_INT = 0
    RED_INT = 1
    BLUE_INT = 2

    def __init__(self, size=11):
        self.size = size
        self.grid = [[self.EMPTY_INT for _ in range(size)] for _ in range(size)]
        self.parent = [i for i in range(size * size + 4)]
        self.rank = [0] * (size * size + 4)
        self.zobrist = ZobristHash(size)
        self.winner = None
        self.TOP_RED = size * size
        self.BOTTOM_RED = size * size + 1
        self.LEFT_BLUE = size * size + 2
        self.RIGHT_BLUE = size * size + 3

    def _colour_to_int(self, colour: Colour) -> int:
        if colour == Colour.RED: return self.RED_INT
        if colour == Colour.BLUE: return self.BLUE_INT
        return self.EMPTY_INT

    def _index(self, row, col):
        return row * self.size + col

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
                if self.rank[root_i] == self.rank[root_j]:
                    self.rank[root_j] += 1

    def play(self, row, col, colour: Colour):
        player_int = self._colour_to_int(colour)
        self.grid[row][col] = player_int
        self.zobrist.update_cell(row, col, colour)

        current_idx = self._index(row, col)
        for k in range(Tile.NEIGHBOUR_COUNT):
            n_row = row + Tile.I_DISPLACEMENTS[k]
            n_col = col + Tile.J_DISPLACEMENTS[k]
            
            if 0 <= n_row < self.size and 0 <= n_col < self.size:
                if self.grid[n_row][n_col] == player_int:
                    self.union(current_idx, self._index(n_row, n_col))

        if colour == Colour.RED:
            if row == 0: self.union(current_idx, self.TOP_RED)
            if row == self.size - 1: self.union(current_idx, self.BOTTOM_RED)
            if self.find(self.TOP_RED) == self.find(self.BOTTOM_RED):
                self.winner = Colour.RED
        elif colour == Colour.BLUE:
            if col == 0: self.union(current_idx, self.LEFT_BLUE)
            if col == self.size - 1: self.union(current_idx, self.RIGHT_BLUE)
            if self.find(self.LEFT_BLUE) == self.find(self.RIGHT_BLUE):
                self.winner = Colour.BLUE

        self.zobrist.update_turn()

    def get_legal_moves(self):
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == self.EMPTY_INT:
                    moves.append((r, c))
        return moves

    def copy(self):
        new_b = Board_Optimized(self.size)
        new_b.grid = [row[:] for row in self.grid]
        new_b.parent = self.parent[:]
        new_b.rank = self.rank[:]
        new_b.winner = self.winner
        new_b.zobrist.hash = self.zobrist.hash
        new_b.zobrist.turn = self.zobrist.turn
        return new_b

    @staticmethod
    def from_game_board(heavy_board):
        opt_board = Board_Optimized(heavy_board.size)
        for r in range(heavy_board.size):
            for c in range(heavy_board.size):
                tile = heavy_board.tiles[r][c]
                if tile.colour in [Colour.RED, Colour.BLUE]:
                    opt_board.play(r, c, tile.colour)
        return opt_board

    def to_nn_input(self, player_perspective: Colour):
        tensor = np.zeros((3, self.size, self.size), dtype=np.float32)
        my_int = self._colour_to_int(player_perspective)
        opp_int = self.BLUE_INT if my_int == self.RED_INT else self.RED_INT
        
        for r in range(self.size):
            for c in range(self.size):
                val = self.grid[r][c]
                if val == my_int:
                    tensor[0][r][c] = 1.0
                elif val == opp_int:
                    tensor[1][r][c] = 1.0
        
        if player_perspective == Colour.BLUE:
            tensor[2, :, :] = 1.0
            
        return tensor