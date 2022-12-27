import os, sys
import random
from colorama import Fore
import json

import numpy as np
import torch


def get_tqdm_config(total, leave=True, color="white"):
    fore_colors = {
        "red": Fore.LIGHTRED_EX,
        "green": Fore.LIGHTGREEN_EX,
        "yellow": Fore.LIGHTYELLOW_EX,
        "blue": Fore.LIGHTBLUE_EX,
        "magenta": Fore.LIGHTMAGENTA_EX,
        "cyan": Fore.LIGHTCYAN_EX,
        "white": Fore.LIGHTWHITE_EX,
    }
    return {
        "file": sys.stdout,
        "total": total,
        "desc": " ",
        "dynamic_ncols": True,
        "bar_format": "{l_bar}%s{bar}%s| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        % (fore_colors[color], Fore.RESET),
        "leave": leave,
    }


def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)


def save_json(path: str, file_name: str, dictionary: dict):
    """Saves dict of floats in json file

    Args:
        path: Folder name you wish to save in
        file_name: The name of file that will be saved as .json
        dictionary: Dictionary you want to save
    """

    PATH = os.path.join(path, file_name)
    if not os.path.exists(path):
        print("Directory does not exist! Making directory {}".format(path))
        os.mkdir(path)
    else:
        print("Directory exists! ")

    with open(PATH, "w", encoding="utf-8") as make_file:
        json.dump(dictionary, make_file, ensure_ascii=False, indent=4)
