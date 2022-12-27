from torchvision.models import ResNet

from .WideResNet import WideResNet
from .ResNet import ResNet18, ResNet34


def create_model(args, device, ema=False):
    # Build WideResNet & EMA model
    if args.model == "WideResNet":
        model = WideResNet(num_classes=10)
        model = model.to(device)

    elif args.model == "ResNet18":
        model = ResNet18(num_classes=10)
        model = model.to(device)

    elif args.model == "ResNet34":
        model = ResNet34(num_classes=10)
        model = model.to(device)

    if ema:
        for param in model.parameters():
            param.detach_()

    return model
