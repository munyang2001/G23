from src.Board import Board
from src.Move import Move
from src.Colour import Colour
from src.Tile import Tile
from collections import deque

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

#Weights
weight_shortest_path = 10
weight_connectivity = 3
weight_bridge = 5
weight_dead_cell = -50

# Shortest path
def shortest_path_length(board: Board, colour: Colour) -> int:
    size = board.size
    INF = 10**9

    dist = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(INF)
        dist.append(row)

    dq = deque()

    # Red connect top to bottom
    if colour == Colour.RED:
        for y in range(size):
            cell = board.tiles[0][y].colour
            if cell == Colour.RED:
                dist[0][y] = 0
                dq.appendleft((0, y))
            elif cell is None:
                dist[0][y] = 1
                dq.append((0, y))

    # Blue connect left to right
    else:
        for x in range(size):
            cell = board.tiles[x][0].colour
            if cell == Colour.BLUE:
                dist[x][0] = 0
                dq.append((x, 0))
            elif cell is None:
                dist[x][0] = 1
                dq.append((x, 0))
    
    # BFS
    while dq:
        x, y = dq.popleft()
        current_cost = dist[x][y]

        # Explore all 6 neighbours
        for k in range(Tile.NEIGHBOUR_COUNT):
            neighbour_x = x + Tile.I_DISPLACEMENTS[k]
            neighbour_y = y + Tile.J_DISPLACEMENTS[k]

            # skip if neighbour is outside the board
            if not (0 <= neighbour_x < size and 0 <= neighbour_y < size):
                continue

            cell = board.tiles[neighbour_x][neighbour_y].colour

            # determine movement cost
            if cell == colour:
                step = 0
            elif cell is None:
                step = 1
            else:
                continue # opponent tile, can't step here

            new_cost = current_cost + step

            if new_cost < dist[neighbour_x][neighbour_y]:
                dist[neighbour_x][neighbour_y] = new_cost
                # 0 cost goes first, 1 cost go later
                if step == 0:
                    dq.appendleft((neighbour_x, neighbour_y))
                else:
                    dq.append((neighbour_x, neighbour_y))

    if colour == Colour.RED:
        best = INF
        for y in range(size):
            value = dist[size - 1][y]
            if value < best:
                best = value
        return best
    else:
        best = INF
        for x in range(size):
            value = dist[x][size - 1]
            if value < best:
                best = value
        return best

# Union-Find connectivity score
def connectivity_score(board: Board, colour: Colour, x: int, y: int) -> int:
    size = board.size
    total = size * size
    uf = UnionFind(total)

    # convert (i,j) to union-find index
    def uf_index(i, j):
        return i * size + j

    # Union all same-colour neighbours on the board
    for i in range(size):
        for j in range(size):
            if board.tiles[i][j].colour != colour:
                continue
            for k in range(Tile.NEIGHBOUR_COUNT):
                neighbour_i = i + Tile.I_DISPLACEMENTS[k]
                neighbour_j = j + Tile.J_DISPLACEMENTS[k]
                if 0 <= neighbour_i < size and 0 <= neighbour_j < size:
                    if board.tiles[neighbour_i][neighbour_j].colour == colour:
                        uf.union(uf_index(i, j), uf_index(neighbour_i, neighbour_j))

    # find how many distinct friendly groups new move (x, y) touches
    touched_roots = set()

    for k in range(Tile.NEIGHBOUR_COUNT):
        neighbour_i = x + Tile.I_DISPLACEMENTS[k]
        neighbour_j = y + Tile.J_DISPLACEMENTS[k]
        if 0 <= neighbour_i < size and 0 <= neighbour_j < size:
            if board.tiles[neighbour_i][neighbour_j].colour == colour:
                touched_roots.add(uf.find(uf_index(neighbour_i, neighbour_j)))

    return len(touched_roots)

# Bridge pattern detection
def is_bridge_move(board:Board, colour: Colour, x: int, y: int) -> bool:
    size = board.size

    bridge_pairs = [
        ((x+1, y),   (x,   y+1)),
        ((x-1, y),   (x,   y-1)),
        ((x+1, y+1), (x,   y-1)),
        ((x+1, y-1), (x,   y+1)),
        ((x-1, y+1), (x,   y)),     # mirrored
        ((x-1, y-1), (x,   y)),
    ]

    for (ax, ay), (bx, by) in bridge_pairs:

        # check if it's outside the board
        if not (0 <= ax < size and 0 <= ay < size):
            continue
        if not (0 <= bx < size and 0 <= by < size):
            continue

        # check if both stones are same colour
        if (board.tiles[ax][ay].colour == colour and board.tiles[bx][by].colour == colour):
            return True

    return False

# Dead cell detection
def is_dead_cell(board: Board, colour: Colour, x: int, y: int) -> bool:
    return False

# Combined score of all heuristics
def heuristic_scoring(board:Board, colour: Colour, x: int, y: int) -> float:
    score = 0

    #shortest path improvement 
    old = shortest_path_length(board, colour)
    board.set_tile_colour(x, y, colour)
    new = shortest_path_length(board, colour)
    board.set_tile_colour(x, y, None)
    score += (old - new) * weight_shortest_path

    # connectivity
    score += connectivity_score(board, colour, x, y) * weight_connectivity

    # bridge
    if is_bridge_move(board, colour, x, y):
        score += weight_bridge

    # dead cell penalty
    if is_dead_cell(board, colour, x, y):
        score += weight_dead_cell

    return score