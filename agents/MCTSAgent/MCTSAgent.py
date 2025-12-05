import subprocess
from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Game import logger

class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._start_process()

    def _start_process(self):
        """Starts the external MCTS agent subprocess."""
        self.agent_process = subprocess.Popen(
            ["./agents/MCTSAgent/mcts-hex"],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Requests a move from the MCTS subprocess based on the current board state."""
        rows = board.tiles
        board_strings = []
        for row in rows:
            row_string = ""
            for tile in row:
                colour = tile.colour
                row_string += "0" if colour is None else colour.get_char()
            board_strings.append(row_string)
        board_string = ",".join(board_strings)

        if opp_move is None:
            command = f"START;;{board_string};{turn};"
        elif opp_move.x == -1 and opp_move.y == -1:
            command = f"SWAP;;{board_string};{turn};"
        else:
            command = f"CHANGE;{opp_move.x},{opp_move.y};{board_string};{turn};"

        # Send command to subprocess
        self.agent_process.stdin.write(command + "\n")
        self.agent_process.stdin.flush()

        # Read response and convert to Move
        response = self.agent_process.stdout.readline().rstrip()
        x, y = map(int, response.split(","))
        return Move(x, y)

    # -------------------------------
    # Deepcopy support
    # -------------------------------
    def __getstate__(self):
        """Remove the unpicklable subprocess before deepcopy."""
        state = self.__dict__.copy()
        state.pop("agent_process", None)
        return state

    def __setstate__(self, state):
        """Restore the state after deepcopy and restart subprocess."""
        self.__dict__.update(state)
        self._start_process()
