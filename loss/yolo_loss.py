import torch

from .utils import get_confidences_tensors, get_coord_tensors, get_class_tensors


class YoloLoss(torch.nn.Module):
    def __init__(
        self,
        S_w: int = 7,
        S_h: int = 7,
        lambda_coord: float = 5,
        lambda_noobj: float = 0.5,
    ) -> None:
        super().__init__()
        self.S_w = S_w
        self.S_h = S_h
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_coord_loss(
        self, predictions: torch.tensor, targets: torch.tensor
    ) -> torch.tensor:
        pred_x_y_coord, pred_w_h_coord, gt_x_y_coord, gt_w_h_coord = get_coord_tensors(
            predictions, targets
        )
        x_y_loss = self.lambda_coord * torch.sum((pred_x_y_coord - gt_x_y_coord) ** 2)
        w_h_loss = self.lambda_coord * torch.sum(
            (torch.sqrt(pred_w_h_coord) - torch.sqrt(gt_w_h_coord)) ** 2
        )
        return x_y_loss + w_h_loss

    def compute_confidence_loss(
        self, predictions: torch.tensor, targets: torch.tensor
    ) -> torch.tensor:
        (
            prediction_oobj_confidence_selector,
            prediction_noobj_confidence_selector,
            target_iou_tensor,
        ) = get_confidences_tensors(predictions, targets)
        oobj_confidences = predictions[prediction_oobj_confidence_selector]
        noobj_confidences = predictions[prediction_noobj_confidence_selector]
        oobj_confidence_loss = torch.sum((oobj_confidences - target_iou_tensor) ** 2)
        noobj_confidence_loss = self.lambda_noobj * torch.sum((noobj_confidences) ** 2)
        return oobj_confidence_loss + noobj_confidence_loss

    def compute_class_loss(
        self, predictions: torch.tensor, targets: torch.tensor
    ) -> torch.tensor:
        prediction_selector, targets_selector = get_class_tensors(prediction=predictions, gt=targets)
        predictions_classes = predictions[prediction_selector]
        target_classes = targets[targets_selector]
        return torch.sum((predictions_classes - target_classes) ** 2)

    def forward(self, predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
        """
        target has shape BATCHxSxSx(5+C), where C=20
        predictions has shape BATCHxSxSx(B*5+C), where C=20, B=2
        """
        coord_loss = self.compute_coord_loss(predictions=predictions, targets=targets)
        confidence_loss = self.compute_confidence_loss(
            predictions=predictions, targets=targets
        )
        class_loss = self.compute_class_loss(predictions=predictions, targets=targets)
        return coord_loss + confidence_loss + class_loss
