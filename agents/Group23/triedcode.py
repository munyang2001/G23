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


import random
import numpy as np
from src.Colour import Colour
from src.Tile import Tile

# ---------------------------
# Zobrist Hashing for Board
# ---------------------------
class ZobristHash:
    table = {}
    turn = 0
    initialized = False

    def __init__(self, board_size=11):
        self.board_size = board_size
        self.hash = 0
        
        # Initialize random values for all cells only once
        if not ZobristHash.initialized:
            ZobristHash.turn = random.getrandbits(64)

            for r in range(board_size):
                for c in range(board_size):
                    val_red = random.getrandbits(64)
                    val_blue = random.getrandbits(64)
                    ZobristHash.table[(r, c, Colour.RED)] = val_red
                    ZobristHash.table[(r, c, Colour.BLUE)] = val_blue

            ZobristHash.initialized = True

    def update_cell(self, row, col, colour):
        """Update hash when a cell is placed or removed."""
        if colour == Colour.RED:
            val = ZobristHash.table[(row, col, Colour.RED)]
            self.hash ^= val
        elif colour == Colour.BLUE:
            val = ZobristHash.table[(row, col, Colour.BLUE)]
            self.hash ^= val

    def update_turn(self):
        """Toggle turn in hash."""
        self.hash ^= ZobristHash.turn

    def get_hash(self):
        return self.hash


# ---------------------------
# Optimized Board Representation
# ---------------------------
class Board_Optimized:
    EMPTY_INT = 0
    RED_INT = 1
    BLUE_INT = 2

    def __init__(self, size=11, zobrist_hash=None):
        self.size = size

        # Initialize empty grid
        self.grid = []
        for r in range(size):
            row = []
            for c in range(size):
                row.append(self.EMPTY_INT)
            self.grid.append(row)

        # Initialize empty spots list
        self.empty_spots = []
        for r in range(size):
            for c in range(size):
                self.empty_spots.append((r, c))

        # Union-find arrays
        self.parent = []
        self.rank = []
        for i in range(size * size + 4):
            self.parent.append(i)
            self.rank.append(0)

        # Zobrist hash object
        if zobrist_hash is not None:
            self.zobrist = zobrist_hash
        else:
            self.zobrist = ZobristHash(size)

        self.winner = None

        # Virtual nodes for RED and BLUE connections
        self.TOP_RED = size * size
        self.BOTTOM_RED = size * size + 1
        self.LEFT_BLUE = size * size + 2
        self.RIGHT_BLUE = size * size + 3

    @staticmethod
    def from_game_board(heavy_board):
        """Convert a standard Board to Board_Optimized."""
        opt_board = Board_Optimized(heavy_board.size)

        for r in range(heavy_board.size):
            for c in range(heavy_board.size):
                tile = heavy_board.tiles[r][c]
                if tile.colour == Colour.RED:
                    opt_board.play(r, c, Colour.RED)
                elif tile.colour == Colour.BLUE:
                    opt_board.play(r, c, Colour.BLUE)

        return opt_board

    def _colour_to_int(self, colour):
        if colour == Colour.RED:
            return self.RED_INT
        elif colour == Colour.BLUE:
            return self.BLUE_INT
        else:
            return self.EMPTY_INT

    def _index(self, row, col):
        return row * self.size + col

    def find(self, i):
        """Union-find 'find' with path compression."""
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """Union by rank."""
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
                if self.rank[root_i] == self.rank[root_j]:
                    self.rank[root_j] += 1

    def play(self, row, col, colour):
        """Place a piece on the board and update state."""
        player_int = self._colour_to_int(colour)
        self.grid[row][col] = player_int

        self.zobrist.update_cell(row, col, colour)
        self.zobrist.update_turn()

        if (row, col) in self.empty_spots:
            self.empty_spots.remove((row, col))

        current_idx = self._index(row, col)

        # Union with neighbors of same colour
        for k in range(Tile.NEIGHBOUR_COUNT):
            n_row = row + Tile.I_DISPLACEMENTS[k]
            n_col = col + Tile.J_DISPLACEMENTS[k]

            if 0 <= n_row < self.size and 0 <= n_col < self.size:
                if self.grid[n_row][n_col] == player_int:
                    neighbor_idx = self._index(n_row, n_col)
                    self.union(current_idx, neighbor_idx)

        # Connect to virtual nodes for RED
        if colour == Colour.RED:
            if row == 0:
                self.union(current_idx, self.TOP_RED)
            if row == self.size - 1:
                self.union(current_idx, self.BOTTOM_RED)

            if self.find(self.TOP_RED) == self.find(self.BOTTOM_RED):
                self.winner = Colour.RED

        # Connect to virtual nodes for BLUE
        if colour == Colour.BLUE:
            if col == 0:
                self.union(current_idx, self.LEFT_BLUE)
            if col == self.size - 1:
                self.union(current_idx, self.RIGHT_BLUE)

            if self.find(self.LEFT_BLUE) == self.find(self.RIGHT_BLUE):
                self.winner = Colour.BLUE

    def get_legal_moves(self):
        """Return a list of all empty cells."""
        return list(self.empty_spots)

    def copy(self):
        """Return a deep copy of the board for simulations."""
        new_hash = ZobristHash(self.size)
        new_hash.hash = self.zobrist.hash

        new_board = Board_Optimized(self.size, zobrist_hash=new_hash)

        new_board.grid = []
        for row in self.grid:
            new_row = list(row)
            new_board.grid.append(new_row)

        new_board.parent = list(self.parent)
        new_board.rank = list(self.rank)
        new_board.winner = self.winner
        new_board.empty_spots = list(self.empty_spots)

        return new_board

    def to_nn_input(self, player_perspective):
        """Convert board to neural network input tensor (3 x size x size)."""
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

        # Plane 3 is all ones if perspective is BLUE
        if player_perspective == Colour.BLUE:
            for r in range(self.size):
                for c in range(self.size):
                    tensor[2][r][c] = 1.0

        return tensor








import random
import math
import time
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from agents.Group23.board_set import Board_Optimized, Tile
import numpy as np

class Node:
    __slots__ = ['parent', 'move', 'player', 'visits', 'wins',
                 'rave_visits', 'rave_wins', 'children', 'allowed_moves']

    def __init__(self, parent=None, move=None, player=None):
        self.parent = parent
        self.move = move
        self.player = player
        self.visits = 0
        self.wins = 0
        self.rave_visits = 0
        self.rave_wins = 0
        self.children = {}
        self.allowed_moves = None

    def is_fully_expanded(self):
        return self.allowed_moves is not None and len(self.allowed_moves) == 0

    def has_children(self):
        return len(self.children) > 0

class MCTS:
    def __init__(self):
        self._C = math.sqrt(2)
        self._RAVE = 300

    def selection(self, board, color, time_limit):
        time_start = time.time()
        root = Node(parent=None, move=None, player=None)
        root.allowed_moves = board.get_legal_moves()

        while time.time() - time_start < time_limit:
            node = root
            board_copy = board.copy()
            red_moves = set()
            blue_moves = set()

            while node.is_fully_expanded() and node.has_children():
                node = self.child_selection(node)
                move_row, move_col = node.move
                move_color = Colour.RED if node.player == 1 else Colour.BLUE
                board_copy.play(move_row, move_col, move_color)
                if node.player == 1:
                    red_moves.add(node.move)
                else:
                    blue_moves.add(node.move)

            if node.allowed_moves and board_copy.winner is None:
                move_to_expand = node.allowed_moves.pop()
                next_player = 1 if node.player is None and color == Colour.RED else (3 - (node.player or 2))
                next_color = Colour.RED if next_player == 1 else Colour.BLUE
                board_copy.play(move_to_expand[0], move_to_expand[1], next_color)

                child_node = Node(parent=node, move=move_to_expand, player=next_player)
                child_node.allowed_moves = board_copy.get_legal_moves()
                node.children[move_to_expand] = child_node
                node = child_node

                rollout_player = 2 if next_player == 1 else 1
            else:
                rollout_player = 1 if (node.player is None and color == Colour.RED) else 3 - node.player

            winner = self.rollout(board_copy, rollout_player, red_moves, blue_moves)
            self.backpropagate(node, winner, red_moves, blue_moves)

        best_child = max(root.children.values(), key=lambda c: c.visits, default=None)
        if best_child is None:
            return random.choice(board.get_legal_moves())
        return best_child.move

    def child_selection(self, node):
        best_score = -float('inf')
        best_node = None
        log_visits = math.log(node.visits) if node.visits > 0 else 0

        for child in node.children.values():
            if child.visits == 0:
                return child
            beta = math.sqrt(self._RAVE / (3 * node.visits + self._RAVE))
            rave_exploitation = child.rave_wins / child.rave_visits if child.rave_visits > 0 else 0
            uct_exploitation = child.wins / child.visits
            exploitation = (1 - beta) * uct_exploitation + beta * rave_exploitation
            exploration = self._C * math.sqrt(log_visits / child.visits)
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

    def rollout(self, board, next_player, red_moves, blue_moves):
        current_player = next_player
        moves = board.get_legal_moves().copy()
        random.shuffle(moves)
        while moves and board.winner is None:
            move = moves.pop()
            colour = Colour.RED if current_player == 1 else Colour.BLUE
            board.play(move[0], move[1], colour)
            if current_player == 1:
                red_moves.add(move)
            else:
                blue_moves.add(move)
            current_player = 3 - current_player
        return board.winner

    def backpropagate(self, node, winner, red_moves, blue_moves):
        moves_by_color = {Colour.RED: red_moves, Colour.BLUE: blue_moves}
        current_player = Colour.RED if node.player == 1 else Colour.BLUE
        reward = 1 if winner == current_player else -1

        while node:
            node.visits += 1
            node.wins += reward
            for child in node.children.values():
                child_color = Colour.RED if child.player == 1 else Colour.BLUE
                if child.move in moves_by_color[child_color]:
                    child.rave_visits += 1
                    if winner == child_color:
                        child.rave_wins += 1
            node = node.parent
            reward = -reward

class Agent(AgentBase):
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.mcts = MCTS()

    def __deepcopy__(self, memo):
        return Agent(self.colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        if turn == 2:
            opp_r, opp_c = -1, -1
            for r in range(self._board_size):
                for c in range(self._board_size):
                    if board.tiles[r][c].colour is not None:
                        opp_r, opp_c = r, c
            if 2 <= opp_r <= 8 and 2 <= opp_c <= 8:
                return Move(-1, -1)

        optimized_board = Board_Optimized.from_game_board(board, self.colour)
        row, col = self.mcts.selection(optimized_board, self.colour, time_limit=9.0)
        return Move(row, col)
