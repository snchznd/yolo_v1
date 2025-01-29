import argparse
from train import add_train_cmd, launch_train_procedure

def parse_args() -> None:
    parser = argparse.ArgumentParser(
        prog="YOLOv1",
        description="Train or do inference with the YOLOv1 model.",
    )
    subparsers = parser.add_subparsers(
        dest="command", title="subcommands", description="subcomands available"
    )

    add_train_cmd(subparsers)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    match args.command:
        case 'train':
            launch_train_procedure(args)
        case _ :
            raise ValueError
