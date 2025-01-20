import torch
from .iou import IoU
from typing import Tuple

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
    idx_selected_bb = None
    selected_bb_iou = None
    if iou_bb_1 >= iou_bb_2:
        idx_selected_bb = 0
        selected_bb_iou = iou_bb_1
    else :
        idx_selected_bb = 1
        selected_bb_iou = iou_bb_2
    return idx_selected_bb, selected_bb_iou


def get_gt_and_pred_bb_selectors(
    prediction: torch.tensor, gt: torch.tensor, S_w: int = 7, S_h: int = 7
) -> Tuple[torch.tensor,torch.tensor]:
    """
    Returns two boolean tensors. The first one is of the same shape as the
    prediction tensor and, when used to extract values on the predictions,
    only selects the bounding boxes (and confidences) that are responsible for
    an object (i.e. there is an object and this bb is responsible for
    predicting it). The second tensor extracts the corresponsing gt bounding
    boxes and confidence values from the gt tensor.
    
    prediction shape : batch x S x S x 30
    gt shape :         batch x S x S x 25
    return shape:      (batch x S x S x 30, batch x S x S x 25)
    """
    batch_size = prediction.shape[0]
    prediction_selector = torch.zeros(
        (batch_size, S_h, S_w, 2 * 5 + 20), dtype=torch.bool
    )
    gt_selector = torch.zeros(
        (batch_size, S_h, S_w, 5 + 20), dtype=torch.bool
    )
    for batch in range(batch_size):
        for cell_x in range(S_w):
            for cell_y in range(S_h):
                # we should only perform this bb selection computation in
                # case there is an object the cell is responsible for !
                if gt[batch, cell_y, cell_x, 4] == 1:
                    # compute IoU to know which bb is responsible for the
                    # object in this cell
                    idx_responsible_bb, _ = get_idx_responsible_bounding_box(
                        prediction=prediction[batch, cell_y, cell_x, ::],
                        ground_truth=gt[batch, cell_y, cell_x, ::],
                        cell_x=cell_x,
                        cell_y=cell_y,
                    )
                    # if the responsible box is 0, we take [x1,y1,w1,h1,c1],
                    # else we take [x2,y2,w2,h2,c2]
                    bb_selector = (
                        slice(5, 10, 1) if idx_responsible_bb else slice(0, 5, 1)
                    )
                    prediction_selector[batch, cell_y, cell_x, bb_selector] = True
                    # if we extract a bb from the predictions in this cell,
                    # then we must also extract the corresponding gt
                    gt_selector[batch, cell_y, cell_x, slice(0,5,1)] = True
    return prediction_selector, gt_selector


def get_coord_tensors(predictions : torch.tensor, targets : torch.tensor) -> Tuple[torch.tensor,torch.tensor,torch.tensor,torch.tensor]:
    prediction_selector, gt_selector = get_gt_and_pred_bb_selectors(prediction=predictions, gt=targets)
    
    selected_prediction_bbs = predictions[prediction_selector]
    selected_gt_bbs = targets[gt_selector]
    #assert selected_prediction_bbs.shape == selected_gt_bbs.shape, 'pred and gt bbs should have same size'
    
    x_y_coord_selector = torch.zeros_like(selected_prediction_bbs, dtype=torch.bool)
    x_y_coord_selector[::5] = True
    x_y_coord_selector[1::5] = True

    w_h_coord_selector = torch.zeros_like(selected_prediction_bbs, dtype=torch.bool)
    w_h_coord_selector[2::5] = True
    w_h_coord_selector[3::5] = True
    
    pred_x_y_coord = selected_prediction_bbs[x_y_coord_selector]
    pred_w_h_coord = selected_prediction_bbs[w_h_coord_selector]
    
    gt_x_y_coord = selected_gt_bbs[x_y_coord_selector]
    gt_w_h_coord = selected_gt_bbs[w_h_coord_selector]
    return pred_x_y_coord, pred_w_h_coord, gt_x_y_coord, gt_w_h_coord


def get_confidences_tensors(
    prediction: torch.tensor, gt: torch.tensor, S_w: int = 7, S_h: int = 7
) -> Tuple[torch.tensor,torch.tensor]:
    """
    prediction shape : batch x S x S x 30
    gt shape :         batch x S x S x 25
    return shape:      (batch x S x S x 30, batch x S x S x 25)
    
    prediction_oobj_confidence_selector: a boolean tensor allowing to select
    the confidence scores corresponding to the responsible boxes (those that
    should thus be trained to match the IoU of the predicted bb and gt bb)
    prediction_noobj_confidence_selector: a boolean tensor allowing to select
    the confidence scores corresponding to the non-responsible boxes (those
    that should thus be trained to equal zero).
    target_iou_tensor: a tensor containing the IoUs of the responsible bounding
    boxes with the gt bounding boxes. This is the target tensor the confidences
    retrieved with the prediction_oobj_confidence_selector boolean selector 
    will be trained to match.
    """
    batch_size = prediction.shape[0]
    prediction_oobj_confidence_selector = torch.zeros(
        (batch_size, S_h, S_w, 2 * 5 + 20), dtype=torch.bool
    )
    prediction_noobj_confidence_selector = torch.zeros(
        (batch_size, S_h, S_w, 2 * 5 + 20), dtype=torch.bool
    )
    target_iou_arr = []
    for batch in range(batch_size):
        for cell_x in range(S_w):
            for cell_y in range(S_h):
                if gt[batch, cell_y, cell_x, 4] == 1:
                    # in case there is an object, we set the oobj selector to
                    # the responsible box and the noobj to the non-responsible
                    idx_responsible_bb, target_iou = get_idx_responsible_bounding_box(
                        prediction=prediction[batch, cell_y, cell_x, ::],
                        ground_truth=gt[batch, cell_y, cell_x, ::],
                        cell_x=cell_x,
                        cell_y=cell_y,
                    )
                    target_iou_arr.append(target_iou)
                    idx_oobj_confidence = 9 if idx_responsible_bb else 4
                    idx_noobj_confidence = (idx_oobj_confidence + 5) % 10
                    prediction_oobj_confidence_selector[batch, cell_y, cell_x, idx_oobj_confidence] = True
                    prediction_noobj_confidence_selector[batch, cell_y, cell_x, idx_noobj_confidence] = True
                else:
                    # in case there is no object, no box is reponsible so both
                    # bb confidences go to the noobj tensor
                    prediction_noobj_confidence_selector[batch, cell_y, cell_x, 4] = True
                    prediction_noobj_confidence_selector[batch, cell_y, cell_x, 9] = True
    target_iou_tensor = torch.tensor(target_iou_arr)#.reshape(batch_size, -1)
    return prediction_oobj_confidence_selector, prediction_noobj_confidence_selector, target_iou_tensor


def get_class_tensors(
    prediction: torch.tensor, gt: torch.tensor, S_w: int = 7, S_h: int = 7
) -> Tuple[torch.tensor,torch.tensor]:
    """
    selects only distributions corresponding to cells containing objects
    
    prediction shape : batch x S x S x 30
    gt shape :         batch x S x S x 25
    return shape:      (batch x S x S x 30, batch x S x S x 25)
    """
    batch_size = prediction.shape[0]
    prediction_selector = torch.zeros(
        (batch_size, S_h, S_w, 2 * 5 + 20), dtype=torch.bool
    )
    gt_selector = torch.zeros(
        (batch_size, S_h, S_w, 5 + 20), dtype=torch.bool
    )
    for batch in range(batch_size):
        for cell_x in range(S_w):
            for cell_y in range(S_h):
                # we only train the class distribution if there is an object
                # in the cell
                if gt[batch, cell_y, cell_x, 4] == 1:
                    prediction_selector[batch, cell_y, cell_x, slice(10,30,1)] = True
                    gt_selector[batch, cell_y, cell_x, slice(5,25,1)] = True
    return prediction_selector, gt_selector
