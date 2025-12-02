import math
import heapq
import time
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Colour import Colour

class ScoutAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.board_size = 11

    @property
    def opponent_colour(self):
        """Dynamic opponent colour to handle Swap rule correctly."""
        return Colour.BLUE if self.colour == Colour.RED else Colour.RED

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Main decision loop.
        """
        # 1. Opening Strategy (If we are Player 1/Red)
        # Play slightly off-center to be "fair" but strong, or classic center.
        # Playing (5,5) is technically the strongest, but (5,4) is harder to swap against.
        # We will stick to the strongest (5,5) - K10 logic.
        if turn == 1:
            return Move(5, 5)

        # 2. Swap Strategy (If we are Player 2/Blue)
        # Rule from your text: Swap if opponent is NOT on the edge.
        if turn == 2 and opp_move is not None:
            if self._should_swap(opp_move):
                return Move(-1, -1)

        # 3. Midgame Strategy (Alpha-Beta Search)
        # We use Iterative Deepening or fixed depth based on game stage.
        # Depth 3 is aggressive but feasible with good move ordering.
        best_move = self._alpha_beta_search(board, depth=2)
        return Move(best_move[0], best_move[1])

    def _should_swap(self, opp_move: Move) -> bool:
        """
        Swap if the move is strictly inside the board (not on the border).
        This follows the heuristic that edge moves are weak, inner moves are strong.
        """
        x, y = opp_move.x, opp_move.y
        # If x or y is 0 or board_size-1, it's an edge move -> Don't Swap.
        is_edge = (x == 0 or x == self.board_size - 1 or 
                   y == 0 or y == self.board_size - 1)
        
        # If it's NOT on the edge, it's strong -> SWAP.
        return not is_edge

    def _alpha_beta_search(self, board, depth):
        legal_moves = self._get_ordered_moves(board)
        if not legal_moves:
            return (-1, -1)

        best_move = legal_moves[0]
        best_val = -math.inf
        alpha = -math.inf
        beta = math.inf

        # Optimisation: Only search top X promising moves to prevent timeout
        moves_to_search = legal_moves[:12]

        for (x, y) in moves_to_search:
            # Execute Move
            tile = board.tiles[x][y]
            tile.colour = self.colour
            
            # Recurse
            val = self._min_value(board, depth - 1, alpha, beta)
            
            # Undo Move
            tile.colour = None

            if val > best_val:
                best_val = val
                best_move = (x, y)
            
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        
        return best_move

    def _min_value(self, board, depth, alpha, beta):
        if depth == 0:
            return self._evaluate_board(board)
        
        val = math.inf
        # Search fewer moves at deeper levels (tapering)
        moves = self._get_ordered_moves(board)[:8]
        
        for (x, y) in moves:
            tile = board.tiles[x][y]
            tile.colour = self.opponent_colour
            
            val = min(val, self._max_value(board, depth - 1, alpha, beta))
            tile.colour = None # Undo
            
            if val <= alpha:
                return val
            beta = min(beta, val)
        return val

    def _max_value(self, board, depth, alpha, beta):
        if depth == 0:
            return self._evaluate_board(board)
        
        val = -math.inf
        moves = self._get_ordered_moves(board)[:8]
        
        for (x, y) in moves:
            tile = board.tiles[x][y]
            tile.colour = self.colour
            
            val = max(val, self._min_value(board, depth - 1, alpha, beta))
            tile.colour = None # Undo
            
            if val >= beta:
                return val
            alpha = max(alpha, val)
        return val

    def _evaluate_board(self, board):
        """
        Evaluation = (Opponent Shortest Path) - (My Shortest Path)
        We use a weighted Dijkstra.
        """
        my_dist = self._dijkstra(board, self.colour)
        opp_dist = self._dijkstra(board, self.opponent_colour)

        # Tactical checks for immediate win/loss
        if my_dist == 0: return 100000
        if opp_dist == 0: return -100000

        # The core heuristic from your text
        return opp_dist - my_dist

    def _dijkstra(self, board, player_colour):
        """
        Dijkstra's Algorithm to find the 'Resistance' or Cost to connect.
        Cost Logic:
        - My Stone: 0
        - Empty: 1 (Resistance)
        - Opponent: Infinity (Blocked)
        """
        pq = []
        visited = set()
        
        # Determine direction
        if player_colour == Colour.RED: # Top -> Bottom (y=0 to y=10)
            target_val = self.board_size - 1
            # Add all viable top row cells
            for x in range(self.board_size):
                tile = board.tiles[x][0]
                cost = self._get_tile_cost(tile, player_colour)
                if cost < 1000:
                    heapq.heappush(pq, (cost, x, 0))
        else: # Blue: Left -> Right (x=0 to x=10)
            target_val = self.board_size - 1
            for y in range(self.board_size):
                tile = board.tiles[0][y]
                cost = self._get_tile_cost(tile, player_colour)
                if cost < 1000:
                    heapq.heappush(pq, (cost, 0, y))

        while pq:
            cost, x, y = heapq.heappop(pq)
            
            if (x, y) in visited: continue
            visited.add((x, y))

            # Check if we reached the other side
            curr_val = y if player_colour == Colour.RED else x
            if curr_val == target_val:
                return cost

            # Hex Neighbors
            neighbors = [
                (x-1, y), (x+1, y), (x, y-1), (x, y+1), (x-1, y+1), (x+1, y-1)
            ]

            for nx, ny in neighbors:
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if (nx, ny) not in visited:
                        tile = board.tiles[nx][ny]
                        move_cost = self._get_tile_cost(tile, player_colour)
                        if move_cost < 1000:
                            heapq.heappush(pq, (cost + move_cost, nx, ny))
        
        return 9999 # No path

    def _get_tile_cost(self, tile, player_colour):
        if tile.colour == player_colour:
            return 0  # No resistance
        elif tile.colour is None:
            return 1  # Standard resistance
        else:
            return 99999 # Infinite resistance (blocked)

    def _get_ordered_moves(self, board):
        """
        Locality Heuristic:
        Instead of searching all empty spots, prioritize spots that are
        adjacent to existing stones (ours or opponents). 
        This is critical for efficiency.
        """
        relevant_moves = set()
        empty_moves = []
        center = self.board_size // 2

        # 1. Identify occupied cells
        occupied = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                tile = board.tiles[x][y]
                if tile.colour is not None:
                    occupied.append((x, y))
                else:
                    # Heuristic: Distance to center
                    dist = abs(x - center) + abs(y - center)
                    empty_moves.append(((x, y), dist))
        
        # 2. Find neighbors of occupied cells (The "Frontier")
        for ox, oy in occupied:
            neighbors = [
                (ox-1, oy), (ox+1, oy), (ox, oy-1), (ox, oy+1), (ox-1, oy+1), (ox+1, oy-1)
            ]
            for nx, ny in neighbors:
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if board.tiles[nx][ny].colour is None:
                        relevant_moves.add((nx, ny))

        # 3. Sort moves
        # Priority A: Neighbors of existing stones (Immediate tactics)
        # Priority B: Center proximity (Strategic)
        
        # Convert set to list
        priority_moves = list(relevant_moves)
        
        # Sort priority moves by closeness to center
        priority_moves.sort(key=lambda m: abs(m[0]-center) + abs(m[1]-center))

        # Sort remaining empty moves
        empty_moves.sort(key=lambda k: k[1])
        sorted_empty = [m[0] for m in empty_moves if m[0] not in relevant_moves]

        # Combine: Frontier moves first, then the rest
        return priority_moves + sorted_empty