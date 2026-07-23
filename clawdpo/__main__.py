import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clawdpo",
        description="Run a ClawDPO training task.",
    )
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
