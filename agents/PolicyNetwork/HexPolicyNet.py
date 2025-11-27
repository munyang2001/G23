import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.PolicyNetwork.Board2Tensor import encode_board_to_tensor
from src.Board import Board
from src.Colour import Colour

class HexPolicyNet(nn.Module):
    def __init__(self, board_size=11, in_channels=4, num_layers=8, width=121):
        super(HexPolicyNet, self).__init__()
        # Hyperparameters
        self.board_size = board_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.width = width

        # The first convolutional layer
        self.conv_first = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.width,
            kernel_size=5,
            padding=2,
            stride=1
        )

        # Intermediate layers
        convs = []
        for _ in range(num_layers-1):
            conv = nn.Conv2d(
                in_channels=self.width,
                out_channels=self.width,
                kernel_size=3,
                padding=1,
                stride=1
            )
            convs.append(conv)
        # Register the convolutional layers for future parameter updates
        self.conv_blocks = nn.ModuleList(convs)

        # A per-position linear combination over the width channels
        self.policy_head = nn.Conv2d(
            in_channels=self.width,
            out_channels=1,
            kernel_size=1,
            padding=0,
            stride=1
        )


    def forward(self,x):
        """
        x: (N, C, H, W)
        Return:
            (N, board_size**2)
        """
        assert x.dim() == 4, f"Expected 4D input (N, C, H, W), got {x.shape}"
        N, C, H, W = x.shape  # batch, channel, height, width.

        assert H == self.board_size and W == self.board_size, (
            f"Expected board size {self.board_size}x{self.board_size}, "
            f"but got {H}x{W}"
        )

        assert self.in_channels == C, (
            f"Expected number of channels: {self.in_channels}, but got {C}"
        )

        # First pass
        x = self.conv_first(x)
        x = F.relu(x)

        # Intermediate passes
        for conv in self.conv_blocks:
            x = conv(x)
            x = F.relu(x)

        # Policy head
        x = self.policy_head(x)

        # Flattens the feature map
        N, C_out, H, W = x.shape
        assert C_out == 1, f"Policy head should produce 1 channel, got {C_out}"

        logits = x.view(N, H * W)

        return logits


    def predict_policy(self, x, legal_mask=None):
        """
        x: input tensor of shape (N, C, H, W)
        legal_mask: optional bool tensor of shape (N, H*W)
            True  -> move is legal
            False -> move is illegal (probability forced to 0)

        Returns:
            probs: tensor of shape (N, H*W), each row sums to 1 over legal moves.
        """
        self.eval()

        with torch.no_grad():
            logits = self.forward(x)

            if legal_mask is not None:
                # Ensure types and shapes match
                assert legal_mask.shape == logits.shape, (
                    f"legal_mask shape {legal_mask.shape} "
                    f"does not match logits shape {logits.shape}"
                )

                minus_inf = torch.tensor(float("-inf"), device=logits.device)
                logits = torch.where(legal_mask, logits, minus_inf)

        probs = F.softmax(logits, dim=-1)

        return probs


def make_legal_mask(all_legal_moves, board_size, batch_size=1, device=None):
    """
    all_legal_moves: list of Move objects for a single board
    board_size: e.g. 11
    Returns: mask of shape (batch_size, board_size * board_size)
    """
    legal_mask = torch.zeros((batch_size, board_size**2), dtype=torch.bool, device=device)

    for move in all_legal_moves:
        idx = move.x * board_size + move.y
        legal_mask[0, idx] = True

    return legal_mask


if __name__ == "__main__":
    # Test:
    fake_board = Board(board_size=11)
    # Put one R stone at (0,0), one B stone at (1,2)
    fake_board.set_tile_colour(0, 0, Colour.RED)
    fake_board.set_tile_colour(1, 2, Colour.BLUE)
    print(fake_board.print_board())

    x = encode_board_to_tensor(fake_board, my_colour=Colour.RED)
    print("Encoded tensor shape:", x.shape)  # expect (1, 4, 11, 11)

    model = HexPolicyNet(board_size=11, in_channels=4)
    logits = model(x)
    print("Logits shape:", logits.shape)  # expect (1, 121)
