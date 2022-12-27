import os, sys
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as transforms


from datasets.build_dataset import get_cifar10
from datasets.data_utils import RandomPadandCrop, RandomFlip, ToTensor
from models.model_utils import create_model
from train import training, evaluate
from train_utils import Loss_Semisupervised, WeightEMA
from parser import MixMatch_parser
from logger import set_logger
from utils import torch_seed, save_json


def run(args):

    # wandb
    wandb.init(
        project="MixMatch with CIFAR10",
        name=args.experiment_dir,
        group=args.model,
        reinit=True,
    )
    wandb.config.update(args)

    # Save path
    experiment_dir = os.path.join("results", args.model, args.experiment_dir)

    # Logger
    _logger = set_logger(os.path.join(experiment_dir, "logging.log"))
    _logger.info(f"Save path: {experiment_dir}")

    # Setting seed and device
    torch_seed(args.seed)
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    _logger.info("Device: {}".format(device))

    # Data preparation
    # Transformation for training data
    transform_train = transforms.Compose(
        [RandomPadandCrop(32), RandomFlip(), ToTensor()]
    )
    _logger.info(f"Finished Transformation Setting")

    # Transformation for valid, test data
    transform_val = transforms.Compose([ToTensor()])

    # Datasets
    train_labeled_set, train_unlabeled_set, val_set, test_set = get_cifar10(
        data_dir="data",
        n_labeled=args.n_labeled,
        n_valid=args.n_valid,
        transform_train=transform_train,
        transform_val=transform_val,
    )
    _logger.info(f"Finished getting Datasets")

    # Dataloaders
    labeled_loader = DataLoader(
        dataset=train_labeled_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    unlabeled_loader = DataLoader(
        dataset=train_unlabeled_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    _logger.info(f"Finished getting Dataloaders: labeled, unlabeled, valid, test")

    # Model
    model = create_model(args, device, ema=False)
    ema_model = create_model(args, device, ema=True)
    _logger.info(f"Finished setting models")

    # Loss functions
    criterion_train = Loss_Semisupervised()
    criterion_val = nn.CrossEntropyLoss().to(device)
    _logger.info(f"Finished setting Loss functions")

    # Optimizers
    optimizer = Adam(model.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, lr=args.lr, alpha=args.ema_decay)
    _logger.info(f"Finished setting Optimizers")

    # Train
    _logger.info(f"Start Training")

    best_loss = np.inf
    # best_loss of validation 기준으로 모멜 저장

    losses, losses_x, losses_u = [], [], []

    train_losses, train_top1s, train_top5s = [], [], []
    val_losses, val_top1s, val_top5s = [], [], []
    test_losses, test_top1s, test_top5s = [], [], []

    for epoch in range(1, args.epochs + 1, 1):
        loss, loss_x, loss_u, model = training(
            args,
            epoch,
            model,
            criterion_train,
            optimizer,
            ema_optimizer,
            labeled_loader,
            unlabeled_loader,
            device,
            wandb,
        )
        losses.append(loss)
        losses_x.append(loss_x)
        losses_u.append(loss_u)

        train_loss, train_top1, train_top5 = evaluate(
            args,
            epoch,
            ema_model,
            criterion_val,
            labeled_loader,
            val_loader,
            test_loader,
            device,
            "Train",
        )
        train_losses.append(train_loss)
        train_top1s.append(train_top1)
        train_top5s.append(train_top5)

        valid_loss, valid_top1, valid_top5 = evaluate(
            args,
            epoch,
            ema_model,
            criterion_val,
            labeled_loader,
            val_loader,
            test_loader,
            device,
            "Valid",
        )
        val_losses.append(valid_loss)
        val_top1s.append(valid_top1)
        val_top5s.append(valid_top5)

        # validation loss 기준 모델 저장
        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(experiment_dir, "best_model.pth"))
            torch.save(ema_model, os.path.join(experiment_dir, "best_ema_model.pth"))

        test_loss, test_top1, test_top5 = evaluate(
            args,
            epoch,
            ema_model,
            criterion_val,
            labeled_loader,
            val_loader,
            test_loader,
            device,
            "Test ",
        )
        test_losses.append(test_loss)
        test_top1s.append(test_top1)
        test_top5s.append(test_top5)

        torch.save(model, os.path.join(experiment_dir, "last_checkpooint_model.pth"))
        torch.save(
            ema_model,
            os.path.join(experiment_dir, "last_checkpoint_ema_model.pth"),
        )

        epoch_result = {
            "Train": {
                "Loss": train_loss,
                "Top1-ACC": train_top1,
                "Top5-ACC": train_top5,
            },
            "Valid": {
                "Loss": valid_loss,
                "Top1-ACC": valid_top1,
                "Top5-ACC": valid_top5,
            },
            "Test": {"Loss": test_loss, "Top1-ACC": test_top1, "Top5-ACC": test_top5},
        }

        _logger.info(
            tabulate(pd.DataFrame(epoch_result).T, headers="keys", tablefmt="psql")
        )


if __name__ == "__main__":
    args = MixMatch_parser().parse_args()
    params = {}
    for key, value in args._get_kwargs():
        params[key] = value

    # Save path
    experiment_dir = os.path.join("results", args.model, args.experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    save_json(experiment_dir, "basic_arguments.json", params)

    # Run
    run(args)
