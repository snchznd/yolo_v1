import argparse
import os
import torch
from yolo.data.utils import load_model, get_class_mapping
from yolo.inference.video_capture import launch_webcam_feed_inference
from yolo.model.yolo_custom import CustomYoloModel

DEFAULT_MODEL_PATH = os.path.expanduser("~/projects/yolo/logs/model/best_model.pth")
DEFAULT_CLASS_MAPPING_PATH = os.path.expanduser(
    "~/projects/yolo/data/classes_mapping.yaml"
)
DEFAULT_RESIZE_FACTOR = 1.5
DEFAULT_STATISTICS_DIR = '~/projects/yolo/data'

def add_webcam_inference_cmd(subparsers: argparse._SubParsersAction) -> None:
    webcam_inference_parser = subparsers.add_parser(
        "webcam_inference", help="Perform live inference on webcam feed."
    )
    webcam_inference_parser.add_argument(
        "-mp",
        "--model_path",
        required=False,
        action="store",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="The path to the model weights that will be used for inference."
    )
    webcam_inference_parser.add_argument(
        "-cmp",
        "--class_mapping_path",
        required=False,
        action="store",
        type=str,
        default=DEFAULT_CLASS_MAPPING_PATH,
        help="The path to the file containing the mapping from indices to class labels."
    )
    webcam_inference_parser.add_argument(
        "-rf",
        "--resize_factor",
        required=False,
        action="store",
        type=float,
        default=DEFAULT_RESIZE_FACTOR,
        help="The scaling factor for the live feed window. Values above 1 degrade performance."
    )
    webcam_inference_parser.add_argument(
        "-sd",
        "--statistics_dir",
        required=False,
        action="store",
        type=str,
        default=DEFAULT_STATISTICS_DIR,
        help="The directory in which the mean and std tensors are contained. Only when using CustomYolo."
    )


def launch_webcam_inference(args: argparse.Namespace) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model_path).to(device).eval()
    if isinstance(model, CustomYoloModel):
        statistics_path = os.path.expanduser(args.statistics_dir)
        statistics_tensor_shape = (1, 3, 1, 1)
        mean = (
            torch.load(os.path.join(statistics_path, "mean.pth"))
            .view(statistics_tensor_shape)
            .to(device)
            .float()
        )
        std = (
            torch.load(os.path.join(statistics_path, "std.pth"))
            .view(statistics_tensor_shape)
            .to(device)
            .float()
        )
    
    class_mapping = get_class_mapping(args.class_mapping_path)
    launch_webcam_feed_inference(
        model=model,
        resize_factor=args.resize_factor,
        classes_mapping=class_mapping,
        mean_tensor=mean,
        std_tensor=std,
    )
