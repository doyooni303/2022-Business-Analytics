import argparse


def MixMatch_parser():
    parser = argparse.ArgumentParser(description="MixMatch PyTorch Implementation")

    parser.add_argument("--experiment_dir", type=str, default=1024)
    parser.add_argument(
        "--model", type=str, default="WideResNet", help="Name of Models"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n_labeled", type=int, default=1024)
    parser.add_argument("--n_valid", type=int, default=100)
    parser.add_argument(
        "--num_iter", type=int, default=1024, help="The number of iteration per epoch"
    )

    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--lambda_u", type=float, default=0.75)
    parser.add_argument("--T", default=0.5, type=float)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--epochs", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)

    return parser
