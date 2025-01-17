import torch
from .utils import get_responsible_bounding_box


class YoloLoss(torch.nn.Module):
    def __init__(
        self,
        S_w: int = 7,
        S_h: int = 7,
        lambda_coord: float = 5,
        lambda_noobj: float = 0.5,
    ) -> None:
        self.S_w = S_w
        self.S_h = S_h
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, target: torch.tensor, gt: torch.tensor):
        """
        target has shape BATCHxSxSx(5+C), where C=20
        gt has shape BATCHxSxSx(B*5+C), where C=20, B=2
        """
        pass
