from tqdm import tqdm

import wandb
import numpy as np
import torch

from utils import get_tqdm_config
from train_utils import interleave, accuracy


def training(
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
):

    losses_t, losses_x, losses_u, ws = 0.0, 0.0, 0.0, 0.0
    model.train()

    # Normally, for loop and DataLoader are used.
    # However, MixMatch dependes on the number of iterations rather than each batch in the data loaders

    iter_labeled = iter(labeled_loader)
    iter_unlabeled = iter(unlabeled_loader)

    with tqdm(**get_tqdm_config(total=args.num_iter, leave=True, color="blue")) as pbar:
        for batch_idx in range(args.num_iter):
            try:
                inputs_x, targets_x = iter_labeled.next()
            except:
                iter_labeled = iter(labeled_loader)
                inputs_x, targets_x = iter_labeled.next()
            real_B = inputs_x.size(0)

            # Transform label to one-hot
            targets_x = torch.zeros(real_B, 10).scatter_(
                1, targets_x.view(-1, 1).long(), 1
            )
            inputs_x, targets_x = inputs_x.to(device), targets_x.to(device)

            try:
                tmp_inputs, _ = iter_unlabeled.next()
            except:
                iter_unlabeled = iter(unlabeled_loader)
                tmp_inputs, _ = iter_unlabeled.next()

            inputs_u1, inputs_u2 = tmp_inputs[0], tmp_inputs[1]
            inputs_u1, inputs_u2 = inputs_u1.to(device), inputs_u2.to(device)

            # Output of unlabeled data
            # : Average of outputs of differently augmented unlabeled data with softmax with Temperature

            with torch.no_grad():
                outputs_u1 = model(inputs_u1)
                outputs_u2 = model(inputs_u2)

                pt = (
                    torch.softmax(outputs_u1, dim=1) + torch.softmax(outputs_u2, dim=1)
                ) / 2
                pt = pt ** (1 / args.T)

                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            # MixUp
            # : Mix inputs and outputs respectively in the same way using convex function

            inputs = torch.cat([inputs_x, inputs_u1, inputs_u2], dim=0)
            targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

            l_mixup = np.random.beta(args.alpha, args.alpha)
            l_mixup = max(l_mixup, 1 - l_mixup)

            # inputs의 index를 섞어 서로 다른 범주끼리 섞도록 하는 역할
            B = inputs.size(0)
            random_idx = torch.randperm(B)

            inputs_a, inputs_b = inputs, inputs[random_idx]
            targets_a, targets_b = targets, targets[random_idx]

            mixed_input = l_mixup * inputs_a + (1 - l_mixup) * inputs_b
            mixed_target = l_mixup * targets_a + (1 - l_mixup) * targets_b

            # (2N, C, H, W) -> (N, C, H, W) & (N, C, H, W)
            # N: Batch size
            # Front: labeled, Back: unlabeled
            mixed_input = list(torch.split(mixed_input, real_B))
            mixed_input = interleave(mixed_input, real_B)

            logits = [model(mixed_input[0])]  # for labeled
            for input in mixed_input[1:]:
                logits.append(model(input))  # for unlabeled

            logits = interleave(logits, real_B)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            loss_x, loss_u, w = criterion_train(
                args,
                logits_x,
                mixed_target[:real_B],
                logits_u,
                mixed_target[real_B:],
                epoch + batch_idx / args.num_iter,
            )

            # Total loss
            loss = loss_x + w * loss_u

            # Backpropagation and Model parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_optimizer.step()

            losses_x += loss_x.item()
            losses_u += loss_u.item()
            losses_t += loss.item()
            ws += w

            log = {
                "Total_loss": losses_t / (batch_idx + 1),
                "Labeled_loss": losses_x / (batch_idx + 1),
                "Unlabeled_loss": losses_u / (batch_idx + 1),
                "W values": ws / (batch_idx + 1),
                "global_step": epoch * args.batch_size + batch_idx,
            }
            wandb.log(log)
            pbar.set_description(
                "[Train(%4d/ %4d)-Total: %.3f|Labeled: %.3f|Unlabeled: %.3f]"
                % (
                    (batch_idx + 1),
                    args.num_iter,
                    losses_t / (batch_idx + 1),
                    losses_x / (batch_idx + 1),
                    losses_u / (batch_idx + 1),
                )
            )
            pbar.update(1)

        pbar.set_description(
            "[Train(%4d/ %4d)-Total: %.3f|Labeled: %.3f|Unlabeled: %.3f]"
            % (
                epoch,
                args.epochs,
                losses_t / (batch_idx + 1),
                losses_x / (batch_idx + 1),
                losses_u / (batch_idx + 1),
            )
        )

    return (
        losses_t / (batch_idx + 1),
        losses_x / (batch_idx + 1),
        losses_u / (batch_idx + 1),
        model,
    )


@torch.no_grad()
def evaluate(
    args,
    epoch,
    ema_model,
    criterion_val,
    labeled_loader,
    val_loader,
    test_loader,
    device,
    phase,
):
    ema_model.eval()

    # Setting Data lodaer by Phase
    if phase == "Train":
        data_loader = labeled_loader
        c = "blue"
    elif phase == "Valid":
        data_loader = val_loader
        c = "green"
    elif phase == "Test ":
        data_loader = test_loader
        c = "red"

    losses = 0.0
    top1s, top5s = [], []

    with tqdm(**get_tqdm_config(total=len(data_loader), leave=True, color=c)) as pbar:
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = ema_model(inputs)
            loss = criterion_val(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses += loss.item()
            top1s.append(prec1)
            top5s.append(prec5)

            log = {
                f"{phase}_loss": losses / (batch_idx + 1),
                f"{phase}_Top1 Acc": np.mean(top1s),
                f"{phase}_Top5 Acc": np.mean(top5s),
                "global_step": epoch * args.batch_size + batch_idx,
            }
            wandb.log(log)

            pbar.set_description(
                "[%s-Loss: %.3f|Top1 Acc: %.3f|Top5 Acc: %.3f]"
                % (phase, losses / (batch_idx + 1), np.mean(top1s), np.mean(top5s))
            )
            pbar.update(1)

        pbar.set_description(
            "[%s(%4d/ %4d)-Loss: %.3f|Top1 Acc: %.3f|Top5 Acc: %.3f]"
            % (
                phase,
                epoch,
                args.epochs,
                losses / (batch_idx + 1),
                np.mean(top1s),
                np.mean(top5s),
            )
        )

    return losses / (batch_idx + 1), np.mean(top1s), np.mean(top5s)
