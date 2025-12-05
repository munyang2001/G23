import math
import time
import random
import numpy as np
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from agents.Group23.board_set import Board_Optimized

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
        self.root = Node(parent=None, move=None, player=None)
        self.transposition_table = {}

    def update_root(self, move):
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = Node(parent=None, move=None, player=None)

    def selection(self, board, color, time_limit):
        time_start = time.time()
        
        if self.root.allowed_moves is None:
            self.root.allowed_moves = board.get_legal_moves()
            if board.hash not in self.transposition_table:
                self.transposition_table[board.hash] = self.root

        if self.root.player is None:
            if color == Colour.RED:
                self.root.player = 1 
            else:
                self.root.player = False

        while time.time() - time_start < time_limit:
            node = self.root
            board_copy = board.copy()
            path = [node]
            path_set = {id(node)}
            cycle_detected = False
            red_moves = set()
            blue_moves = set()

            while node.is_fully_expanded() and node.has_children():
                node = self.child_selection(node)
                
                if id(node) in path_set:
                    cycle_detected = True
                    break
                
                path.append(node)
                path_set.add(id(node))
                
                move_row, move_col = node.move
                move_color = Colour.RED if node.player == 1 else Colour.BLUE
                board_copy.play(move_row, move_col, move_color)
                
                if node.player == 1:
                    red_moves.add(node.move)
                else:
                    blue_moves.add(node.move)

            if cycle_detected:
                continue

            if node.allowed_moves and board_copy.winner is None:
                move_to_expand = node.allowed_moves.pop()
                next_player = 1 if node.player is None and color == Colour.RED else (3 - (node.player or 2))
                next_color = Colour.RED if next_player == 1 else Colour.BLUE
                
                board_copy.play(move_to_expand[0], move_to_expand[1], next_color)
                board_hash = board_copy.hash

                if board_hash in self.transposition_table:
                    child_node = self.transposition_table[board_hash]
                    if move_to_expand not in node.children:
                        node.children[move_to_expand] = child_node
                else:
                    child_node = Node(parent=node, move=move_to_expand, player=next_player)
                    child_node.allowed_moves = board_copy.get_legal_moves()
                    self.transposition_table[board_hash] = child_node
                    node.children[move_to_expand] = child_node

                node = child_node
                path.append(node)
                rollout_player = 2 if next_player == 1 else 1
            else:
                rollout_player = 1 if (node.player is None and color == Colour.RED) else 3 - node.player

            winner = self.rollout(board_copy, rollout_player, red_moves, blue_moves)
            self.backpropagate(path, winner, red_moves, blue_moves)

        best_child = max(self.root.children.values(), key=lambda c: c.visits, default=None)
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

    def backpropagate(self, path, winner, red_moves, blue_moves):
        moves_by_color = {Colour.RED: red_moves, Colour.BLUE: blue_moves}
        
        for node in reversed(path):
            current_player = Colour.RED if node.player == 1 else Colour.BLUE
            reward = 1 if winner == current_player else -1
            
            node.visits += 1
            node.wins += reward
            
            for child in node.children.values():
                child_color = Colour.RED if child.player == 1 else Colour.BLUE
                if child.move in moves_by_color[child_color]:
                    child.rave_visits += 1
                    if winner == child_color:
                        child.rave_wins += 1

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

        if opp_move is not None:
            opp_move_tuple = (opp_move._x, opp_move._y)
            self.mcts.update_root(opp_move_tuple)

        optimized_board = Board_Optimized.from_game_board(board, self.colour)
        row, col = self.mcts.selection(optimized_board, self.colour, time_limit=8.5)
        my_move_tuple = (row, col)
        self.mcts.update_root(my_move_tuple)
        return Move(row, col)