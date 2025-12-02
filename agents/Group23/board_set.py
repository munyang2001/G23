import numpy as np
from src.Colour import Colour
from src.Tile import Tile
import random



ZOBRIST_TABLE = [
    [[random.getrandbits(128) for _ in range(2)] for _ in range(11)]
    for _ in range(11)
]

TURN_HASH = random.getrandbits(128)



class Board_Optimized:
    RED_INT = 1
    BLUE_INT = 2
    EMPTY = 0

    def __init__(self, player, size=11):
        self.size = size
        self.turn = player
        self.grid = np.zeros((size, size), dtype=int)
        self.empty_spots = np.array(list(np.ndindex(size, size)))
        self.winner = None

        self.parent = np.arange(size * size + 4)
        self.rank = np.zeros(size * size + 4, dtype=int)

        self.TOP_RED = size * size
        self.BOTTOM_RED = size * size + 1
        self.LEFT_BLUE = size * size + 2
        self.RIGHT_BLUE = size * size + 3
        self.hash = 0
        if player == Colour.RED:
            self.hash ^= TURN_HASH

    @staticmethod
    def from_game_board(heavy_board, player):
        opt_board = Board_Optimized(player, heavy_board.size)
        for r in range(heavy_board.size):
            for c in range(heavy_board.size):
                tile = heavy_board.tiles[r][c]
                if tile.colour == Colour.RED:
                    opt_board.play(r, c, Colour.RED)
                elif tile.colour == Colour.BLUE:
                    opt_board.play(r, c, Colour.BLUE)
        return opt_board

    def _index(self, row, col):
        return row * self.size + col

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        ri = self.find(i)
        rj = self.find(j)
        if ri == rj:
            return
        if self.rank[ri] > self.rank[rj]:
            self.parent[rj] = ri
        elif self.rank[rj] > self.rank[ri]:
            self.parent[ri] = rj
        else:
            self.parent[rj] = ri
            self.rank[ri] += 1

    def play(self, row, col, colour):
        colour_int = self.RED_INT if colour == Colour.RED else self.BLUE_INT
        self.grid[row, col] = colour_int
        current = self._index(row, col)

        if colour == Colour.RED:
            hash_id = 0 
        else:
            hash_id =1
        self.hash ^= ZOBRIST_TABLE[row][col][hash_id]
        self.hash ^= TURN_HASH
        for k in range(Tile.NEIGHBOUR_COUNT):
            nr = row + Tile.I_DISPLACEMENTS[k]
            nc = col + Tile.J_DISPLACEMENTS[k]
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self.grid[nr, nc] == colour_int:
                    neighbor = self._index(nr, nc)
                    self.union(current, neighbor)

        if colour == Colour.RED:
            if row == 0: self.union(current, self.TOP_RED)
            if row == self.size - 1: self.union(current, self.BOTTOM_RED)
            if self.find(self.TOP_RED) == self.find(self.BOTTOM_RED):
                self.winner = Colour.RED

        if colour == Colour.BLUE:
            if col == 0: self.union(current, self.LEFT_BLUE)
            if col == self.size - 1: self.union(current, self.RIGHT_BLUE)
            if self.find(self.LEFT_BLUE) == self.find(self.RIGHT_BLUE):
                self.winner = Colour.BLUE

    def get_legal_moves(self):
        return [tuple(pos) for pos in self.empty_spots if self.grid[pos[0], pos[1]] == self.EMPTY]

    def copy(self):
        new_board = Board_Optimized(self.turn, self.size)
        new_board.grid = np.copy(self.grid)
        new_board.parent = np.copy(self.parent)
        new_board.rank = np.copy(self.rank)
        new_board.winner = self.winner
        new_board.empty_spots = np.copy(self.empty_spots)
        new_board.hash = self.hash
        return new_board

    def to_nn_input(self, player_perspective):
        tensor = np.zeros((3, self.size, self.size), dtype=np.float32)
        my_int = self.RED_INT if player_perspective == Colour.RED else self.BLUE_INT
        opp_int = self.BLUE_INT if my_int == self.RED_INT else self.RED_INT

        tensor[0][self.grid == my_int] = 1.0
        tensor[1][self.grid == opp_int] = 1.0
        if player_perspective == Colour.BLUE:
            tensor[2, :, :] = 1.0
        return tensor
