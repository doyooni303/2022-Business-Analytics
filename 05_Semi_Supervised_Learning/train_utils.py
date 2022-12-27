import numpy as np
import torch
import torch.nn.functional as F


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class WeightEMA(object):  # EMA=Exponential Moving Average
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model

        self.alpha = alpha

        self.params = list(self.model.state_dict().items())
        self.ema_params = list(self.ema_model.state_dict().items())

        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param[1].data.copy_(ema_param[1].data)

    def step(self):
        inverse_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param[1].dtype == torch.float32:
                ema_param[1].mul_(
                    self.alpha
                )  # ema_params_new = self.alpha * ema_params_old
                ema_param[1].add_(
                    param[1] * inverse_alpha
                )  # ema_params_Double_new = (1-self.alpha)*params

                # summary: ema_params_new = self.alpha*ema_params_old + (1-self.alpha)*params
                # params: parameters of training model
                param[1].mul_(1 - self.wd)


class Loss_Semisupervised(object):
    def __call__(self, args, outputs_x, target_x, outputs_u, targets_u, epoch):
        self.args = args
        probs_u = torch.softmax(outputs_u, dim=1)

        loss_x = -torch.mean(
            torch.sum(F.log_softmax(outputs_x, dim=1) * target_x, dim=1)
        )

        # L2 loss for unlabeled data
        loss_u = torch.mean((probs_u - targets_u) ** 2)

        return (
            loss_x,
            loss_u,
            self.args.lambda_u * linear_rampup(epoch, self.args.epochs),
        )


def interleave_offsets(batch_size, nu):
    groups = [batch_size // (nu + 1)] * (nu + 1)
    for x in range(batch_size - sum(groups)):
        groups[-x - 1] += 1

    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)

    assert offsets[-1] == batch_size
    return offsets


def interleave(xy, batch_size):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch_size, nu)

    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if k == 1:
            correct_k = correct[:k].view(-1).float().sum(0)
        if k > 1:
            correct_k = correct[:k].float().sum(0).sum(0)
        acc = correct_k.mul_(100.0 / batch_size)
        acc = acc.detach().cpu().numpy()
        res.append(acc)
    return res
