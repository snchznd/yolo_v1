import matplotlib.axes
import matplotlib.patches as patches
import matplotlib
import torch
from matplotlib import pyplot as plt
import yaml
from typing import Dict, Optional

from yolo.loss.utils import un_normalize_bounding_box
from yolo.loss.iou import IoU
from yolo.model.yolo import YoloModel

BEST_MODEL_PATH = '/home/masn/projects/yolo/logs/model/best_model.pth'
CLASSES_MAPPING_PATH = '/home/masn/projects/yolo/classes_mapping.yaml'

def get_yolo_bounding_box_corner_coordinates(
    bounding_box: torch.tensor,
) -> torch.tensor:
    bb_center_x, bb_center_y, bb_width, bb_height = bounding_box
    bb_corner_x = bb_center_x - bb_width / 2
    bb_corner_y = bb_center_y - bb_height / 2
    return bb_corner_x.item(), bb_corner_y.item(), bb_width.item(), bb_height.item()


def plot_model_prediction(img : torch.tensor,
                          model : torch.nn.Module,
                          classes_mapping: Optional[Dict] = None,
                          confidence_treshold : float = .5,
                          iou_treshold : float = .5,
                          S_w : int = 7,
                          S_h : int = 7) -> None:
    img = img.to('cuda')
    model.eval()
    img_reshaped = torch.unsqueeze(img, 0)
    prediction = model(img_reshaped)
    prediction = torch.squeeze(prediction,0) 
        
    fig, ax = plt.subplots()
    ax.imshow(img.cpu().permute(1, 2, 0).int())
    
    bbs_to_plot = []
    for cell_x in range(S_w):
        for cell_y in range(S_h):
            bb_1 = prediction[cell_y, cell_x,0:4]
            bb_1_conf = prediction[cell_y, cell_x,4]
            bb_2 = prediction[cell_y, cell_x,5:9]
            bb_2_conf = prediction[cell_y, cell_x,9]
            bbs_to_add = []
            if bb_1_conf > confidence_treshold and bb_2_conf < confidence_treshold:
                bbs_to_add.append(bb_1)
            elif bb_1_conf < confidence_treshold and bb_2_conf > confidence_treshold:
                bbs_to_add.append(bb_2)
            elif bb_1_conf > confidence_treshold and bb_2_conf > confidence_treshold:
                iou = IoU(*bb_1, *bb_2)
                if iou > iou_treshold:
                    # IoU between bb is too high. They are likely representing
                    # the same object. Take only the bb with the max confidence
                    bb_to_plot = bb_1 if bb_1_conf > bb_2_conf else bb_2
                    bbs_to_add.append(bb_to_plot)
                    # un-normalize this bb and plot it 
                else:
                    # IoU between bb is low, they are likey representing different
                    # objects. Plot both.
                    bbs_to_add.append(bb_1)
                    bbs_to_add.append(bb_2)
            for bb in bbs_to_add:
                # 1. un-normalize bb
                un_normalized_bb = un_normalize_bounding_box(bounding_box=bb, cell_x=cell_x, cell_y=cell_y)
                bbs_to_plot.append(un_normalized_bb)
                
            if classes_mapping and bbs_to_add:
                predicted_class_idx = torch.argmax(prediction[cell_y,cell_x,10:30]).item()
                predicted_class = classes_mapping[predicted_class_idx + 1]
                print(f'[x={cell_x},y={cell_y}]: {predicted_class}')
    
    for bb in bbs_to_plot:
        x, y, w, h = get_yolo_bounding_box_corner_coordinates(
            bb.detach().clone()
        )
        rect = patches.Rectangle(
            xy=(x, y), width=w, height=h, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
    plt.show()
    
def load_model(weights_path : str = BEST_MODEL_PATH, device='cuda') -> YoloModel:
    model = YoloModel()
    model_weights = torch.load(weights_path)
    model.load_state_dict(model_weights)
    model.to(device)
    return model

def get_class_mapping(file_path : str = CLASSES_MAPPING_PATH) -> Dict:
    with open(file=file_path, encoding='utf-8', mode='r') as f:
        classes_dict = yaml.safe_load(f)
    return classes_dict