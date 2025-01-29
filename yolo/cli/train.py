import argparse


def add_train_cmd(subparsers: argparse._SubParsersAction) -> None:
    train_parser = subparsers.add_parser("train", help="Launch a training session.")
    train_parser.add_argument(
        "-lbm",
        "--load_best_model",
        required=False,
        action="store_true",
        default=False,
        help="Whether to load the previous best model or start from zero.",
    )
    train_parser.add_argument(
        "--config",
        required=True,
        action="store",
        type=str,
        help="The path to the config file.",
    )

def launch_train_procedure(args : argparse.Namespace) -> None:
    pass