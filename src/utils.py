"""Utility functions."""
import argparse
import os
import gc
import subprocess
import pickle
import sys
import collections
import psutil
import math
import numpy as np
import torch
from logger import create_logger

FALSY_STRINGS = {'off', 'false', '0', 'no'}
TRUTHY_STRINGS = {'on', 'true', '1', 'yes'}


def ln2float(module):
    """Batchnorm to Float."""
    if isinstance(module, torch.nn.LayerNorm):
        print('Warning: Casting LayerNorm to fp32 ...')
        module.float()
    for child in module.children():
        ln2float(child)
    return module


def bool_flag(s):
    """Parse boolean arguments from the command line."""
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def initialize_exp(args, logger_filename='train.log'):
    """Initialize the experiment.

    - dump argparse arguments
    - setup python logging
    - set the random seeds
    """
    args.parent_dump_path = f"../{args.dump_path}"
    args.dump_path = os.path.join(args.parent_dump_path, args.exp_name)

    if not os.path.isdir(args.parent_dump_path):
        print(f"making {args.parent_dump_path}...")
        subprocess.Popen("mkdir %s" % args.parent_dump_path, shell=True).wait()

    if not os.path.isdir(args.dump_path):
        print(f"making {args.dump_path}...")
        subprocess.Popen("mkdir %s" % args.dump_path, shell=True).wait()

    pickle.dump(
        args,
        open(os.path.join(args.dump_path, 'args.pkl'), 'wb')
    )

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            command.append("'%s'" % x)
    args.command = ' '.join(command)

    # random seed
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # create a logger
    logger = create_logger(os.path.join(args.dump_path, logger_filename))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(args)).items())))
    logger.info('The experiment will be stored in %s\n' % args.dump_path)
    logger.info('Running command: %s' % args.command)
    return logger

