import torch
from loss.iou import IoU

def plot_model_prediction(img : torch.tensor,
                          model : torch.nn.Module,
                          confidence_treshold : float = .5,
                          iou_treshold : float = .5,
                          S_w : int = 7,
                          S_h : int = 7) -> None:
    img.reshape((-1, 3, 448, 448))
    model.eval()
    prediction = model(img)
    prediction.reshape((S_h, S_w, 30))
    
    # plot image here...
    
    for cell_x in range(S_w):
        for cell_y in range(S_h):
            bb_1 = prediction[cell_y, cell_x,0:4]
            bb_1_conf = prediction[cell_y, cell_x,4]
            bb_2 = prediction[cell_y, cell_x,5:9]
            bb_2_conf = prediction[cell_y, cell_x,9]
            bbs_to_plot = []
            if bb_1_conf > confidence_treshold and bb_2_conf < confidence_treshold:
                bbs_to_plot.append(bb_1)
            elif bb_1_conf < confidence_treshold and bb_2_conf > confidence_treshold:
                bbs_to_plot.append(bb_2)
            elif bb_1_conf > confidence_treshold and bb_2_conf > confidence_treshold:
                iou = IoU(*bb_1, *bb_2)
                if iou > iou_treshold:
                    # IoU between bb is too high. They are likely representing
                    # the same object. Take only the bb with the max confidence
                    bb_to_plot = bb_1 if bb_1_conf > bb_2_conf else bb_2
                    bbs_to_plot.append(bb_to_plot)
                    # un-normalize this bb and plot it 
                else:
                    # IoU between bb is low, they are likey representing different
                    # objects. Plot both.
                    bbs_to_plot.append(bb_1)
                    bbs_to_plot.append(bb_2)
            for bb in bbs_to_plot:
                # 1. un-normalize bb
                # 2. plot bb
                pass
    