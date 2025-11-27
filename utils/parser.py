import argparse
import torch
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='predict')
    parser.add_argument('--year', type=int, default=2021)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args


def dir_init(default_args):
    from copy import deepcopy
    """ args 받은다음, device, Home directory, data_dir, log_dir, output_dir, 들 지정하고, Path들 체크해서  """
    args = deepcopy(default_args)
    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        args.home = os.path.dirname(os.path.dirname(__file__))
        print(args.home)
    elif sysChecker() == "Windows":
        args.home = ''
        # args.batch_size, args.num_epochs = 4, 2
        # args.debug = True
        pass
    else:
        raise Exception("Check Your Platform Setting (Linux-Server or Windows)")
    # Check path
    return args