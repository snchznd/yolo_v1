import torch
from .iou import IoU

PREDICTION_BB_1_COORDINATES = slice(0, 4, 1)
PREDICTION_BB_2_COORDINATES = slice(5, 9, 1)
TARGET_BB_COORDINATES = slice(0, 4, 1)


def un_normalize_bounding_box(
    bounding_box: torch.tensor,
    cell_x: int,
    cell_y: int,
    image_width: int = 448,
    image_height: int = 448,
    S_w: int = 7,
    S_h: int = 7,
) -> torch.tensor:
    """
    bounding_box is a torch.tensor of length 4 of format [x,y,w,h]
    x and y are scaled w.r.t. the cell dimension
    w and h are sclaed w.r.t. the image size
    """
    scaled_bounding_box = torch.zeros_like(bounding_box)

    # scaling w and h back to absolute values
    scaled_bounding_box[2] = bounding_box[2] * image_width
    scaled_bounding_box[3] = bounding_box[3] * image_height

    # scaling x and y back to absolute values
    cell_width = image_width // S_w
    cell_height = image_height // S_h
    scaled_bounding_box[0] = (cell_x + bounding_box[0]) * cell_width
    scaled_bounding_box[1] = (cell_y + bounding_box[1]) * cell_height

    return scaled_bounding_box


def get_idx_responsible_bounding_box(
    prediction: torch.tensor,
    ground_truth: torch.tensor,
    cell_x: int,
    cell_y: int,
    image_width: int = 448,
    image_height: int = 448,
    S_w: int = 7,
    S_h: int = 7,
) -> int:
    """
    prediction is of shape 2*5+C and ground_truth is of shape 5+C
    [x1,y1,w1,h1,c1,x2,y2,w2,h2,c2,p1,...,p20]
    """
    bb_pred_1 = un_normalize_bounding_box(
        bounding_box=prediction[PREDICTION_BB_1_COORDINATES],
        cell_x=cell_x,
        cell_y=cell_y,
        image_width=image_width,
        image_height=image_height,
        S_w=S_w,
        S_h=S_h,
    )
    bb_pred_2 = un_normalize_bounding_box(
        bounding_box=prediction[PREDICTION_BB_2_COORDINATES],
        cell_x=cell_x,
        cell_y=cell_y,
        image_width=image_width,
        image_height=image_height,
        S_w=S_w,
        S_h=S_h,
    )
    bb_target = un_normalize_bounding_box(
        bounding_box=ground_truth[TARGET_BB_COORDINATES],
        cell_x=cell_x,
        cell_y=cell_y,
        image_width=image_width,
        image_height=image_height,
        S_w=S_w,
        S_h=S_h,
    )

    iou_bb_1 = IoU(*bb_pred_1, *bb_target)
    iou_bb_2 = IoU(*bb_pred_2, *bb_target)

    # return index of BB with biggest IoU
    return 0 if iou_bb_1 >= iou_bb_2 else 1


def get_responsible_bb_selector(
    prediction: torch.tensor, gt: torch.tensor, S_w: int = 7, S_h: int = 7
) -> torch.tensor:
    """
    extract only the responsible bounding box
    prediction shape : batch x S x S x 30
    gt shape :         batch x S x S x 25
    return shape:      batch x S x S x 30
    """
    batch_size = prediction.shape[0]
    prediction_selector = torch.zeros(
        (batch_size, S_h, S_w, 2 * 5 + 20), dtype=torch.bool
    )
    for batch in range(batch_size):
        for cell_x in range(S_w):
            for cell_y in range(S_h):
                # we should only perform this bb selection computation in
                # case there is an object the cell is responsible for !
                if gt[batch, cell_y, cell_x, 4] == 1:
                    idx_responsible_bb = get_idx_responsible_bounding_box(
                        prediction=prediction[batch, cell_y, cell_x, ::],
                        ground_truth=gt[batch, cell_y, cell_x, ::],
                        cell_x=cell_x,
                        cell_y=cell_y,
                    )
                    bb_selector = (
                        slice(0, 5, 1) if idx_responsible_bb else slice(5, 10, 1)
                    )
                    prediction_selector[batch, cell_y, cell_x, bb_selector] = True
    return prediction_selector
