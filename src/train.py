"""Main run script."""
import os
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
import pickle
import time
import json
import copy

try:
    from apex.fp16_utils import FP16_Optimizer
except ModuleNotFoundError:
    print(f"WARNING: apex module not found. Looking in {os.path.dirname(os.getcwd())}...")
    sys.path.append(f'{os.path.dirname(os.getcwd())}/apex')
from apex.fp16_utils import FP16_Optimizer

from data import MultiBPTTIterator, MultiBPTTAutoIterator
from opt import AdamCosineWithWarmup
from utils import bool_flag, initialize_exp, ln2float
from model import TransformerLM

parser = argparse.ArgumentParser()
# Experiment related arguments
parser.add_argument(
    "--dump_path", type=str, required=True,
    help="Experiment dump path"
)
parser.add_argument(
    "--exp_name", type=str, default="",
    help="Experiment name"
)
parser.add_argument(
    "--exp_id", type=str, default="",
    help="Experiment ID"
)
parser.add_argument(
    "--seed", type=int, default=-1,
    help="Random generator seed (-1 for random)"
)
parser.add_argument(
    "--reload", type=bool_flag,
    help="Reload experiment from dump path if exists"
)

# Model arguments
parser.add_argument(
    "--emb_dim", type=int, default=768,
    help="Embedding layer size"
)
parser.add_argument(
    "--n_layers", type=int, default=12,
    help="Number of layers in the encoders"
)
parser.add_argument(
    "--dropout", type=float, default=0.1,
    help="Dropout"
)
parser.add_argument(
    "--transformer_ffn_emb_dim", type=int, default=3072,
    help="Transformer fully-connected hidden dim size"
)
parser.add_argument(
    "--attention_dropout", type=float, default=0.1,
    help="attention_dropout"
)
parser.add_argument(
    "--relu_dropout", type=float, default=0.1,
    help="relu_dropout"
)
parser.add_argument(
    "--attention_heads", type=int, default=12,
    help="encoder_attention_heads"
)
parser.add_argument(
    "--activation", type=str, default='relu',
    help="Type of activation to use"
)
parser.add_argument(
    "--fp16", type=bool_flag,
    help="Train with fp16."
)
parser.add_argument(
    "--freeze_emb", type=bool_flag,
    help="freeze embeddings"
)

# Data related arguments
parser.add_argument(
    "--cache", type=bool_flag,
    help="Recompute or used cached vocab"
)
parser.add_argument(
    "--max_length", type=int, default=512,
    help="Maximum length of sentences (after BPE)"
)
parser.add_argument(
    "--max_vocab", type=int, default=-1,
    help="Maximum vocabulary size (-1 to disable)"
)
parser.add_argument(
    "--batch_size", type=int, default=64,
    help="Batch size"
)
parser.add_argument(
    "--buffer_size", type=int, default=4e6,
    help="Number of sentences too keep in the buffer."
)

# Optimizer related arguments
parser.add_argument(
    "--n_warmup_steps", type=int, default=10000,
    help="Number of warmup steps for Adam"
)
parser.add_argument(
    "--n_grad_accumulation_steps", type=int, default=1,
    help="Number steps to accumulate gradients before updating params"
)

# Training loop related arguments
parser.add_argument(
    "--n_epochs", type=int, default=30,
    help="Maximum epoch size"
)
parser.add_argument(
    "--print_freq", type=int, default=3200,
    help="Print Frequency"
)
parser.add_argument(
    "--eval_freq", type=int, default=1280000,
    help="Evaluation/Save Frequency"
)

# train & dev info
parser.add_argument(
    "--train_dict", type=str, required=True,
    help="Dict of the paths to the training files (json)"
)
parser.add_argument(
    "--dev_dict", type=str, required=True,
    help="Dict of the paths to the dev files (json)"
)

# pretraining / finetunning arguments
parser.add_argument(
    "--datasets", type=str, required=True,
    help="List of datasets to train on. curriculum separated by ',' and mixes separated by '+'"
)
parser.add_argument(
    "--patience", type=int, default=20,
    help="number of validation step not improved before switching to finetunning"
)

# Adapative params
parser.add_argument(
    "--adaptive_inputs", type=bool_flag,
    help="Use adaptive input embeddings"
)
parser.add_argument(
    "--adaptive_softmax", type=bool_flag,
    help="Adaptive softmax for speedup and memory"
)


def reload_model(model, model_ckpt_path):
    """Reload a model."""
    model_load = torch.load(model_ckpt_path)
    model.load_state_dict(model_load)


def reload_optimizer(optimizer, opt_ckpt_path):
    """Reload an optimizer."""
    opt_load = torch.load(opt_ckpt_path)
    optimizer.load_state_dict(opt_load)


def load_iterators():
    train_info = json.loads(params.train_dict)
    for v in train_info.values():
        v['fname'] = f"../{v['fname']}"
        v['vocab_size'] = params.max_vocab
        v['bptt_length'] = params.max_length
        v['cache'] = params.cache
        v['cache_dir'] = params.parent_dump_path
    dev_info = json.loads(params.dev_dict.replace('\'', '\"'))
    for v in dev_info.values():
        v['fname'] = f"../{v['fname']}"
        v['vocab_size'] = -1
        v['bptt_length'] = params.max_length
        v['cache'] = True
        v['cache_dir'] = params.parent_dump_path

    train_iterator = MultiBPTTAutoIterator(
        info_dict=train_info,
        cache_dir=params.parent_dump_path,
        cache=params.cache
    )
    dev_iterator = MultiBPTTIterator(
        info_dict=dev_info,
        cache_dir=params.parent_dump_path,
        cache=True
    )
    # Use the same vocab learned on the train set for the dev set as well.
    dev_iterator.word2id = train_iterator.multi_iterator.word2id
    dev_iterator.id2word = train_iterator.multi_iterator.id2word
    dev_iterator.freeze_limit = train_iterator.multi_iterator.freeze_limit
    for itrt in dev_iterator.iterators.values():
        itrt.word2id = train_iterator.multi_iterator.word2id
        itrt.id2word = train_iterator.multi_iterator.id2word
        itrt.freeze_limit = train_iterator.multi_iterator.freeze_limit

    return train_iterator, dev_iterator


def build_model(word2id, freeze_limit):
    model = TransformerLM(
        n_words=len(word2id), emb_dim=params.emb_dim,
        n_layers=params.n_layers, dropout=params.dropout,
        pad_index=word2id['<pad>'],
        attention_heads=params.attention_heads,
        attention_dropout=params.attention_dropout,
        relu_dropout=params.relu_dropout,
        ffn_dim=params.transformer_ffn_emb_dim,
        freeze_emb=params.freeze_emb,
        freeze_limit=freeze_limit,
        activation=params.activation,
        adaptive_softmax=params.adaptive_softmax,
        adaptive_inputs=params.adaptive_inputs
    )
    logger.info(model)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info('Model has %d parameters' % num_params)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model has %d trainable parameters' % num_params)
    return model


def build_optimizer(parameters):
    if params.fp16:
        # Use apex's FP 16 optimizer for mixed precison and to do loss scaling
        optimizer = FP16_Optimizer(
            AdamCosineWithWarmup(
                parameters, betas=(0.9, 0.98),
                eps=1e-6, weight_decay=0.01
            ),
            dynamic_loss_scale=True
        )
    else:
        optimizer = AdamCosineWithWarmup(
            parameters, betas=(0.9, 0.98),
            eps=1e-6, weight_decay=0.01
        )
    return optimizer


def one_batch(train_iterator, dataset, model, optimizer):
    (inp, output), is_last = train_iterator.get_minibatch(dataset, params.batch_size)
    # Move minibatch to GPU
    if torch.cuda.is_available():
        inp = inp.cuda()
        output = output.cuda()

    # Average loss computed by GPU
    loss = model(inp, output)
    loss = loss.mean() / params.n_grad_accumulation_steps

    # Loss computation semantics based on apex FP16 loss scaling
    if params.fp16:
        optimizer.backward(loss)
    else:
        loss.backward()
    return loss.item(), is_last


def validation(dev_iterator, dataset, model):
    model.eval()
    dev_iterator.fetch_buffer(dataset)
    dev_buffer_size = dev_iterator.iterators[dataset].buffer_tensor.size(0)
    dev_losses = []
    with torch.no_grad():
        for i in range(0, dev_buffer_size, params.batch_size):
            inp, output = dev_iterator.get_minibatch(dataset, i, params.batch_size)
            if torch.cuda.is_available():
                inp = inp.cuda()
                output = output.cuda()

            loss = model(inp, output)
            loss = loss.mean()
            dev_losses.append(loss.item())
    return dev_losses


def save_best_model(model, optimizer, best_dev_ce):
    f_model = os.path.join(params.dump_path, 'model.pt')
    f_opt = os.path.join(params.dump_path, 'optimizer.pt')
    f_meta = os.path.join(params.dump_path, 'meta.pkl')
    torch.save(model.state_dict(), f_model)
    torch.save(optimizer.state_dict(), f_opt)
    n_steps = optimizer.optimizer.n_current_steps if params.fp16 else optimizer.n_current_steps
    pickle.dump(
        {
            'best_dev_ce': best_dev_ce,
            'n_steps': n_steps,
        },
        open(f_meta, 'wb')
    )


def save_curr_model(model, optimizer, datasets, patience):
    f_model = os.path.join(params.dump_path, 'cur_model.pt')
    f_opt = os.path.join(params.dump_path, 'cur_optimizer.pt')
    f_meta2 = os.path.join(params.dump_path, 'meta2.json')
    torch.save(model.state_dict(), f_model)
    torch.save(optimizer.state_dict(), f_opt)
    json.dump(
        {
            "patience": patience,  # this NEEDS to be saved regardless of model improvement
            "datasets": datasets  # same thing for this
        },
        open(f_meta2, 'w')
    )


def main():
    train_iterator, dev_iterator = load_iterators()

    model = build_model(train_iterator.multi_iterator.word2id,
                        train_iterator.multi_iterator.freeze_limit)

    # Mask the pad token when computing losses
    loss_fn = nn.CrossEntropyLoss(
        reduction='mean',
        ignore_index=train_iterator.multi_iterator.word2id['<pad>']
    )

    assert params.n_gpus > 0
    logger.info('Using %d GPUs' % (torch.cuda.device_count()))

    # Give model the loss function so loss can be computed with DataParallel
    model.loss_fn = loss_fn

    model = nn.DataParallel(
        model,
        device_ids=list(range(torch.cuda.device_count()))
    )

    if torch.cuda.is_available() and params.fp16:
        assert torch.backends.cudnn.enabled
        # Convert params to fp16 except BN params.
        model = model.half().cuda()
        model = ln2float(model)

    elif torch.cuda.is_available():
        assert torch.backends.cudnn.enabled
        model = model.cuda()

    optimizer = build_optimizer(model.parameters())

    best_dev_ce = 99999.
    patience = params.patience  # initial patience

    # build initial dataset distribution to use
    datasets = {}
    train_info = json.loads(params.train_dict)
    for data_name in params.datasets.split('+'):
        datasets[data_name] = train_info[data_name].get('finetune_weight', -1)
    # replace all weights '-1' by the proba available
    taken = sum([p for p in datasets.values() if p > 0])  # proba already assigned
    no_weights = len([p for p in datasets.values() if p < 0])  # number of datasets with no proba
    remain = 1.0 - taken
    # some data have no sample proba and current proba don't sum to 1: split the available mass equally
    if no_weights > 0 and remain > 0:
        for d, w in datasets.items():
            if w < 0:
                datasets[d] = remain / no_weights
    # all datasets have proba mass but they don't sum to 1: add equal proba mass to each
    elif no_weights == 0 and remain > 0:
        n = len(datasets)
        for d, w in datasets.items():
            datasets[d] += remain / n
    # in all other cases, raise error for now...
    else:
        raise ValueError(f"Something wrong with the current probability masses in {datasets}")
    assert sum(datasets.values()) == 1.0, "data weights must sum to 1! %s" % datasets.values()

    if params.reload:
        # Reload the model from the best checkpoint so far.
        # update best_dev_ce, patience, and datasets accordingly

        # Check if model and optimizer exist
        model_exists = os.path.exists(os.path.join(params.dump_path, 'model.pt'))
        optimizer_exists = os.path.exists(os.path.join(params.dump_path, 'optimizer.pt'))
        meta_exists = os.path.exists(os.path.join(params.dump_path, 'meta.pkl'))
        meta2_exists = os.path.exists(os.path.join(params.dump_path, 'meta2.json'))

        if not model_exists:
            logger.info('Asked to reload model but model.pt not found in %s ' % params.dump_path)
        else:
            logger.info('Reloading model')
            reload_model(model, os.path.join(params.dump_path, 'model.pt'))

        if not optimizer_exists:
            logger.info('Asked to reload optimizer but optimizer.pt not found in %s ' % params.dump_path)
        else:
            logger.info('Reloading optimizer')
            reload_optimizer(optimizer, os.path.join(params.dump_path, 'optimizer.pt'))

        if not meta_exists:
            logger.info('Asked to reload metadata but meta.pkl not found in %s ' % params.dump_path)
        else:
            logger.info('Reloading metadata')
            meta = pickle.load(
                open(os.path.join(params.dump_path, 'meta.pkl'), 'rb')
            )
            best_dev_ce = meta['best_dev_ce']
            if params.fp16:
                optimizer.optimizer.n_current_steps = meta['n_steps']
            else:
                optimizer.n_current_steps = meta['n_steps']

        if not meta2_exists:
            logger.info('Asked to reload metadata but meta2.json not found in %s ' % params.dump_path)
        else:
            logger.info('Reloading metadata 2')
            meta2 = json.load(
                open(os.path.join(params.dump_path, 'meta2.json'), 'r')
            )
            patience = meta2['patience']
            datasets = meta2['datasets']

    ###########################
    # TRAINING
    ###########################
    logger.info("Datasets: %s" % datasets)

    # if p>0 at this step, training is not done
    if patience > 0:
        patience, datasets = train(model, datasets, train_iterator, dev_iterator, optimizer, patience, best_dev_ce)

    # if p>0 again here, then training stopped before the end...
    if patience > 0:
        logger.warning("Stopped training before the end..? patience: %d" % patience)

    logger.info('=' * 80)

    return


def train(model, datasets, train_iterator, dev_iterator, optimizer, patience, best_dev_ce):
    start_time = time.time()
    losses = []
    examples_processed = 0
    total_examples_processed = 0
    not_improved = 0  # counter for the number of epochs it didn't improve on validation
    d2epochs = {d: 0 for d in datasets.keys()}

    # load the first buffer
    for d in datasets.keys():
        train_iterator.fetch_buffer(d)

    torch.cuda.empty_cache()
    try:
        for _ in range(params.n_epochs):
            # Counter for grad aggregation
            ctr = 0
            # flag to know if the previously sampled dataset reset its buffer
            was_last = {d: True for d in datasets.keys()}
            while True:
                # sample a dataset for this batch
                dataset = np.random.choice(list(datasets.keys()), p=list(datasets.values()))
                if was_last[dataset]:
                    d2epochs[dataset] += 1
                    logger.info('Training on buffer %d for dataset %s' % (d2epochs[dataset], dataset))
                    logger.info('Sampled datasets {}'.format(d2epochs))
                epoch = d2epochs[dataset]

                loss, is_last = one_batch(train_iterator, dataset, model, optimizer)
                # update was_last flag
                was_last[dataset] = is_last

                losses.append(loss * params.n_grad_accumulation_steps)

                examples_processed += params.batch_size
                total_examples_processed += params.batch_size

                # Update params every `n_grad_accumulation_steps` times
                if ctr % params.n_grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                ctr += 1

                if total_examples_processed % params.print_freq == 0 and total_examples_processed > 0:
                    # Mean of the last few time steps
                    cur_loss = np.mean(losses)
                    time_elapsed = time.time() - start_time
                    ms_per_example = (time_elapsed / float(examples_processed)) * 1000.
                    cur_lr = optimizer.optimizer.cur_lr if params.fp16 else optimizer.cur_lr
                    logger.info('-' * 80)
                    logger.info(
                        'Buffer # {:3d} | {:5d} examples processed | '
                        'ms/sequence {:5.2f} | lr {:5.8f} | data {} | loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, total_examples_processed,
                            ms_per_example, cur_lr, '+'.join(datasets.keys()),
                            cur_loss, np.exp(cur_loss)
                        )
                    )
                    logger.info('-' * 80)
                    examples_processed = 0
                    start_time = time.time()
                    losses = []

                if total_examples_processed % params.eval_freq == 0 and total_examples_processed > 0:
                    dev_ce = 0
                    for d, w in datasets.items():
                        logger.info('Computing Valid set PPL on %s...' % d)
                        dev_losses = validation(dev_iterator, d, model)
                        tmp_dev_ce = np.nanmean(dev_losses)
                        logger.info('{:5.3f}'.format(np.exp(tmp_dev_ce)))
                        dev_ce += (w * tmp_dev_ce)

                    not_improved += 1
                    patience -= 1
                    if dev_ce < best_dev_ce:
                        # reset counters
                        best_dev_ce = dev_ce
                        not_improved = 0
                        patience = params.patience

                        logger.info('Saving best model at %.5f PPL ...' % np.exp(best_dev_ce))
                        save_best_model(model, optimizer, best_dev_ce)

                    logger.info('-' * 80)
                    logger.info(
                        'Buffer # {:3d} | {:5d} examples processed | data {} | Valid CE {:5.3f} |'
                        ' Valid PPL {:5.3f} | Improved {:3d} steps ago | patience {:3d}'.format(
                            epoch, total_examples_processed, datasets,
                            dev_ce, np.exp(dev_ce), not_improved, patience
                        )
                    )
                    logger.info('-' * 80)

                    logger.info('Saving current model ...')
                    save_curr_model(model, optimizer, datasets, patience)

                    model.train()

                    if patience < 0:
                        return patience, datasets

        return patience, datasets
    except KeyboardInterrupt:
        return patience, datasets


if __name__ == '__main__':
    params = parser.parse_args()
    logger = initialize_exp(params)

    # Use all available GPUs. To use fewer, use CUDA_VISIBLE_DEVICES at launch
    params.n_gpus = torch.cuda.device_count()

    main()
