"""Main run script."""
import argparse
import json
import time
import sys
import os

parser = argparse.ArgumentParser(description='Launcher script')
# Experiment related arguments
parser.add_argument(
    "--config", type=str, required=True,
    help="Config JSON file path"
)
parser.add_argument(
    "--dataset", type=str, required=True,
    help="Dataset name"
)
parser.add_argument(
    "--experiment_name", type=str, required=True,
    help="Experiment name for model dump path and logs"
)
parser.add_argument(
    "--test", action="store_true",
    help="Just prints out the command going to be run"
)

params = parser.parse_args()

# Dataset to paths map
dataset2paths = {
    # no proofs + facts
    'clutrr1_no_proof_facts_2': {
        "dump_path": "models/1_no-proof_facts_2",
        "dataset_train": "data/forward/train/no_proof_1.2_train_facts_anon.txt.4000",
        "dataset_valid": "data/forward/valid/no_proof_1.2_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_no_proof_facts_4': {
        "dump_path": "models/1_no-proof_facts_4",
        "dataset_train": "data/forward/train/no_proof_1.4_train_facts_anon.txt.4000",
        "dataset_valid": "data/forward/valid/no_proof_1.4_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_no_proof_facts_6': {
        "dump_path": "models/1_no-proof_facts_6",
        "dataset_train": "data/forward/train/no_proof_1.6_train_facts_anon.txt.4000",
        "dataset_valid": "data/forward/valid/no_proof_1.6_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    # long proofs + facts
    'clutrr1_long_proof_facts_2': {
        "dump_path": "models/1_long-proof_facts_2",
        "dataset_train": "data/forward/train/long_proof_1.2_train_facts_anon.txt.4000",
        "dataset_valid": "data/forward/valid/long_proof_1.2_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_long_proof_facts_4': {
        "dump_path": "models/1_long-proof_facts_4",
        "dataset_train": "data/forward/train/long_proof_1.4_train_facts_anon.txt.4000",
        "dataset_valid": "data/forward/valid/long_proof_1.4_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_long_proof_facts_6': {
        "dump_path": "models/1_long-proof_facts_6",
        "dataset_train": "data/forward/train/long_proof_1.6_train_facts_anon.txt.4000",
        "dataset_valid": "data/forward/valid/long_proof_1.6_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    # long proofs reversed + facts
    'clutrr1_long-proof-rev_facts_2': {
        "dump_path": "models/1_long-proof-rev_facts_2",
        "dataset_train": "data/backward/train/long_proof_1.2_train_facts_anon.txt.4000",
        "dataset_valid": "data/backward/valid/long_proof_1.2_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_long-proof-rev_facts_4': {
        "dump_path": "models/1_long-proof-rev_facts_4",
        "dataset_train": "data/backward/train/long_proof_1.4_train_facts_anon.txt.4000",
        "dataset_valid": "data/backward/valid/long_proof_1.4_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_long-proof-rev_facts_6': {
        "dump_path": "models/1_long-proof-rev_facts_6",
        "dataset_train": "data/backward/train/long_proof_1.6_train_facts_anon.txt.4000",
        "dataset_valid": "data/backward/valid/long_proof_1.6_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    # short proofs + facts
    'clutrr1_short_proof_facts_2': {
        "dump_path": "models/1_short-proof_facts_2",
        "dataset_train": "data/forward/train/short_proof_1.2_train_facts_anon.txt.4000",
        "dataset_valid": "data/forward/valid/short_proof_1.2_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_short_proof_facts_4': {
        "dump_path": "models/1_short-proof_facts_4",
        "dataset_train": "data/forward/train/short_proof_1.4_train_facts_anon.txt.4000",
        "dataset_valid": "data/forward/valid/short_proof_1.4_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_short_proof_facts_6': {
        "dump_path": "models/1_short-proof_facts_6",
        "dataset_train": "data/forward/train/short_proof_1.6_train_facts_anon.txt.4000",
        "dataset_valid": "data/forward/valid/short_proof_1.6_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    # short proofs reversed + facts
    'clutrr1_short-proof-rev_facts_2': {
        "dump_path": "models/1_short-proof-rev_facts_2",
        "dataset_train": "data/backward/train/short_proof_1.2_train_facts_anon.txt.4000",
        "dataset_valid": "data/backward/valid/short_proof_1.2_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_short-proof-rev_facts_4': {
        "dump_path": "models/1_short-proof-rev_facts_4",
        "dataset_train": "data/backward/train/short_proof_1.4_train_facts_anon.txt.4000",
        "dataset_valid": "data/backward/valid/short_proof_1.4_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_short-proof-rev_facts_6': {
        "dump_path": "models/1_short-proof-rev_facts_6",
        "dataset_train": "data/backward/train/short_proof_1.6_train_facts_anon.txt.4000",
        "dataset_valid": "data/backward/valid/short_proof_1.6_valid_facts_anon.txt.4000",
        "buffer_size": -1,
    },
    # no proof + amt
    'clutrr1_no_proof_amt_2': {
        "dump_path": "models/1_no-proof_amt_2",
        "dataset_train": "data/forward/train/no_proof_1.2_train_amt_anon.txt.4000",
        "dataset_valid": "data/forward/valid/no_proof_1.2_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_no_proof_amt_4': {
        "dump_path": "models/1_no-proof_amt_4",
        "dataset_train": "data/forward/train/no_proof_1.4_train_amt_anon.txt.4000",
        "dataset_valid": "data/forward/valid/no_proof_1.4_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_no_proof_amt_6': {
        "dump_path": "models/1_no-proof_amt_6",
        "dataset_train": "data/forward/train/no_proof_1.6_train_amt_anon.txt.4000",
        "dataset_valid": "data/forward/valid/no_proof_1.6_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    # long proof + amt
    'clutrr1_long_proof_amt_2': {
        "dump_path": "models/1_long-proof_amt_2",
        "dataset_train": "data/forward/train/long_proof_1.2_train_amt_anon.txt.4000",
        "dataset_valid": "data/forward/valid/long_proof_1.2_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_long_proof_amt_4': {
        "dump_path": "models/1_long-proof_amt_4",
        "dataset_train": "data/forward/train/long_proof_1.4_train_amt_anon.txt.4000",
        "dataset_valid": "data/forward/valid/long_proof_1.4_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_long_proof_amt_6': {
        "dump_path": "models/1_long-proof_amt_6",
        "dataset_train": "data/forward/train/long_proof_1.6_train_amt_anon.txt.4000",
        "dataset_valid": "data/forward/valid/long_proof_1.6_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    # long proofs reversed + amt
    'clutrr1_long-proof-rev_amt_2': {
        "dump_path": "models/1_long-proof-rev_amt_2",
        "dataset_train": "data/backward/train/long_proof_1.2_train_amt_anon.txt.4000",
        "dataset_valid": "data/backward/valid/long_proof_1.2_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_long-proof-rev_amt_4': {
        "dump_path": "models/1_long-proof-rev_amt_4",
        "dataset_train": "data/backward/train/long_proof_1.4_train_amt_anon.txt.4000",
        "dataset_valid": "data/backward/valid/long_proof_1.4_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_long-proof-rev_amt_6': {
        "dump_path": "models/1_long-proof-rev_amt_6",
        "dataset_train": "data/backward/train/long_proof_1.6_train_amt_anon.txt.4000",
        "dataset_valid": "data/backward/valid/long_proof_1.6_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    # short proof + amt
    'clutrr1_short_proof_amt_2': {
        "dump_path": "models/1_short-proof_amt_2",
        "dataset_train": "data/forward/train/short_proof_1.2_train_amt_anon.txt.4000",
        "dataset_valid": "data/forward/valid/short_proof_1.2_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_short_proof_amt_4': {
        "dump_path": "models/1_short-proof_amt_4",
        "dataset_train": "data/forward/train/short_proof_1.4_train_amt_anon.txt.4000",
        "dataset_valid": "data/forward/valid/short_proof_1.4_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_short_proof_amt_6': {
        "dump_path": "models/1_short-proof_amt_6",
        "dataset_train": "data/forward/train/short_proof_1.6_train_amt_anon.txt.4000",
        "dataset_valid": "data/forward/valid/short_proof_1.6_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    # short proofs reversed + amt
    'clutrr1_short-proof-rev_amt_2': {
        "dump_path": "models/1_short-proof-rev_amt_2",
        "dataset_train": "data/backward/train/short_proof_1.2_train_amt_anon.txt.4000",
        "dataset_valid": "data/backward/valid/short_proof_1.2_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_short-proof-rev_amt_4': {
        "dump_path": "models/1_short-proof-rev_amt_4",
        "dataset_train": "data/backward/train/short_proof_1.4_train_amt_anon.txt.4000",
        "dataset_valid": "data/backward/valid/short_proof_1.4_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
    'clutrr1_short-proof-rev_amt_6': {
        "dump_path": "models/1_short-proof-rev_amt_6",
        "dataset_train": "data/backward/train/short_proof_1.6_train_amt_anon.txt.4000",
        "dataset_valid": "data/backward/valid/short_proof_1.6_valid_amt_anon.txt.4000",
        "buffer_size": -1,
    },
}

# parse datasets and build train_dict & valid_dict
datasets = params.dataset.split('+')  # mixed datasets separated by '+'
train_dict = {
    d: {
        "fname": dataset2paths[d]['dataset_train'],
        "buffer_size": dataset2paths[d]['buffer_size'],
    }
    for d in datasets
}
dev_dict = {
    d: {
        "fname": dataset2paths[d]['dataset_valid'],
        "buffer_size": -1
    }
    for d in datasets
}

if len(datasets) > 1:
    # build a custom path from the set of paths for each dataset
    dump_paths = [dataset2paths[d]['dump_path'] for d in datasets]
    dump_path = []  # the actual dump path that we will join with '_'
    for dp in dump_paths:
        items = dp.split('_')  # split each paths around '_'
        dump_path.extend([e for e in items if e not in dump_path])  # add every new items to the dump path
    dump_path = '_'.join(dump_path)
else:
    dump_path = dataset2paths[params.dataset]['dump_path']

######################
# run script
######################
# NOTE: should be run with the following hardware:
# --gpu 6 \
# --cpu 12 \
# --mem 128 \
# --gpu-mem 32 \
python_run_script = f'cd src/ && python train.py ' \
                    f'--exp_name {params.experiment_name} ' \
                    f'--dump_path {dump_path} ' \
                    f'--train_dict \'{json.dumps(train_dict)}\' ' \
                    f'--dev_dict \'{json.dumps(dev_dict)}\' ' \
                    f'--datasets {params.dataset} '

# Add hyperparameters to run string.
run_config = json.load(open(params.config, 'r'))
for k, v in run_config.items():
    python_run_script += f' --{k} {str(v)}'

# name = params.experiment_name + '_' + params.dataset
# python_run_script += f" > ../logs/log_{name}.log 2>&1"

if params.test:
    print("\npython script: '%s'" % python_run_script)
    sys.exit(1)
else:
    # Dump python run script in to a shell script file.
    os.system(python_run_script)
