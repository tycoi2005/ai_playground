# main.py

import argparse
from dino_control import run_dino_controller

def main():
    parser = argparse.ArgumentParser(description="Run Dino Jump Controller using hand or pose gestures.")
    parser.add_argument(
        "--mode",
        choices=["hand", "pose"],
        default="hand",
        help="Choose input mode: 'hand' for hand gesture, 'pose' for body pose detection (default: hand)."
    )
    args = parser.parse_args()

    run_dino_controller(mode=args.mode)

if __name__ == "__main__":
    main()
