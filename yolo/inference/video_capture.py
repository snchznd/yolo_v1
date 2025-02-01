from typing import Dict, List, Optional, Tuple

import time
import cv2 as cv
import torch
import torchvision.transforms.v2
import torchvision.tv_tensors

from yolo.loss.iou import IoU
from yolo.loss.utils import un_normalize_bounding_box
from yolo.model.yolo import YoloModel
from yolo.data.utils import get_class_mapping


def get_bbs_to_plot(
    prediction: torch.tensor,
    classes_mapping: Optional[Dict] = None,
    confidence_treshold: float = 0.1,
    iou_treshold: float = 0.5,
    S_w: int = 7,
    S_h: int = 7,
) -> List[torch.tensor]:
    prediction = torch.squeeze(prediction, 0)

    predicted_classes = []
    bbs_to_plot = []
    for cell_x in range(S_w):
        for cell_y in range(S_h):
            bb_1 = prediction[cell_y, cell_x, 0:4]
            bb_1_conf = prediction[cell_y, cell_x, 4]
            bb_2 = prediction[cell_y, cell_x, 5:9]
            bb_2_conf = prediction[cell_y, cell_x, 9]
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
                un_normalized_bb = un_normalize_bounding_box(
                    bounding_box=bb, cell_x=cell_x, cell_y=cell_y
                )
                bbs_to_plot.append(un_normalized_bb)

            if classes_mapping and bbs_to_add:
                predicted_class_idx = torch.argmax(
                    prediction[cell_y, cell_x, 10:30]
                ).item()
                predicted_class = classes_mapping[predicted_class_idx + 1]
                predicted_classes.append(predicted_class)
    return bbs_to_plot, predicted_classes


def get_top_right_bottom_left_coord(bounding_box: torch.tensor) -> torch.tensor:

    coords = torch.zeros_like(bounding_box)
    coords = []

    bb_center_x, bb_center_y, bb_width, bb_height = bounding_box

    coords.append(bb_center_x - bb_width / 2)
    coords.append(bb_center_y - bb_height / 2)

    coords.append(bb_center_x + bb_width / 2)
    coords.append(bb_center_y + bb_height / 2)
    return coords


def launch_webcam_feed_inference(
    model: YoloModel,
    resize_factor: float = 2,
    confidence_treshold: float = 0.3,
    classes_mapping: Optional[Dict] = None,
) -> None:
    model.eval()
    cap = cv.VideoCapture(-1)
    input_resize_transform = torchvision.transforms.v2.Resize(size=(448,) * 2)

    original_frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    original_frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    print(f"{original_frame_width=}")
    print(f"{original_frame_height=}")

    cap.set(cv.CAP_PROP_FRAME_WIDTH, original_frame_width * resize_factor)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, original_frame_height * resize_factor)

    frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    print(f"{frame_width=}")
    print(f"{frame_height=}")

    output_resize_transform = torchvision.transforms.v2.Resize(
        size=(frame_height, frame_width)
    )
    prev_frame_time = time.perf_counter()
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        img = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
        img = input_resize_transform(img).to("cuda")
        pred = model(img)
        bbs_to_plot, predicted_classes = get_bbs_to_plot(
            prediction=pred,
            confidence_treshold=confidence_treshold,
            classes_mapping=classes_mapping,
        )
        if bbs_to_plot:
            bbs_corners = [get_top_right_bottom_left_coord(bb) for bb in bbs_to_plot]
            # print(bbs_corners)
            bounding_boxes = torchvision.tv_tensors.BoundingBoxes(
                data=bbs_corners,
                format=torchvision.tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(448, 448),  # H, W
            )

            resized_bbs = output_resize_transform(bounding_boxes)

            # draw on the frame
            for idx, bb in enumerate(resized_bbs):
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = bb
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = (
                    int(top_left_x),
                    int(top_left_y),
                    int(bottom_right_x),
                    int(bottom_right_y),
                )
                cv.rectangle(
                    frame,
                    (top_left_x, top_left_y),
                    (bottom_right_x, bottom_right_y),
                    (0, 0, 255),
                    6,
                )

                if classes_mapping and predicted_classes and idx < len(predicted_classes):
                    text_to_display = predicted_classes[idx]
                    cv.putText(
                        frame,
                        text_to_display,
                        (top_left_x, top_left_y - 10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        3,
                    )

        current_frame_time = time.perf_counter()
        fps = int(1 / (current_frame_time - prev_frame_time))
        prev_frame_time = current_frame_time
        cv.putText(
            frame,
            str(fps),
            (7, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 0, 0),
            3,
            cv.LINE_AA,
        )

        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    import os

    from yolo.data.utils import load_model

    weights_path = os.path.expanduser("~/projects/yolo/logs/model/best_model.pth")
    model = load_model(weights_path).to("cuda").eval()
    class_mapping = get_class_mapping(
        "/home/masn/projects/yolo/data/classes_mapping.yaml"
    )
    RESIZE_FACTOR = 2
    CONFIDENCE_TRESHOLD = 0.275
    launch_webcam_feed_inference(
        model=model,
        resize_factor=RESIZE_FACTOR,
        confidence_treshold=CONFIDENCE_TRESHOLD,
        classes_mapping=class_mapping,
    )
