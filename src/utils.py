import argparse


def init_run() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--save_ckpt", action="store_true")
    return parser.parse_args()
