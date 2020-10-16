"""Generate from the model."""
import os
import re
import gc
import copy
import torch
import torch.nn as nn
import argparse
import pickle
import time
import collections
import fastBPE
import numpy as np
from sacremoses import MosesTokenizer, MosesDetokenizer

from utils import undo_bpe
from model import TransformerLM


def get_args():
    parser = argparse.ArgumentParser(description='Sample from a trained GPT-2 Model')
    parser.add_argument(
        "--load_path", type=str, required=True,
        help="Path to the folder that contains a saved model"
    )
    parser.add_argument(
        "--bpe_codes", type=str, required=True,
        help="Path to BPE codes gotten by fastBPE"
    )
    parser.add_argument(
        "--bpe_vocab_path", type=str, required=True,
        help="Path to BPE Vocab gotten by fastBPE"
    )
    parser.add_argument(
        "--gpt_vocab_path", type=str, required=True,
        help="Path to GPT pickle vocab file"
    )
    parser.add_argument(
        '--test_file', type=str, required=True,
        help="Path to the test file to read and condition on"
    )
    parser.add_argument(
        '--out_file', type=str, required=True,
        help="Path to the file to write the predictions on"
    )

    # sampling params
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Softmax temperature"
    )
    parser.add_argument(
        "--max_length", type=int, default=256,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--sample", type=str, default='topk',
        help="Sampling strategy (topk, nucleus, temperature, greedy)"
    )

    # other
    parser.add_argument(
        "--max_batch_size", type=int, default=128,
        help="Maximum number of sentences to generate in a batch"
    )
    parser.add_argument(
        "--verbose", type=str, default='0',
        help="be verbose"
    )
    parser.add_argument(
        "--save_frequency", type=int, default=5,
        help="number of hours between each file saved"
    )

    return parser.parse_args()


def order_by_length(test_lines):
    """
    Order a list of string by their number of words
    :param test_lines: list of BPEncoded strings
    :return: ordered list and the original indices to put the list back in its original order
    """
    if not isinstance(test_lines, np.ndarray):
        test_lines = np.array(test_lines)
    # reordering the test lines by increasing length
    all_test_lengths = list(map(lambda line: len(line.split()), test_lines))
    idx_sorted_lengths = np.argsort(all_test_lengths)
    idx_original_lengths = np.argsort(idx_sorted_lengths)
    # test_lines.sort(key=lambda line: len(line.split()))
    test_lines = test_lines[idx_sorted_lengths]
    return test_lines, idx_original_lengths


def build_size_batches(test_lines):
    """
    making list of batches of sequences of equal lengths
    :param test_lines: list of strings (already BPEncoded)
    :return: list of batches. each batch containing a collection of same-length-strings.
    """
    lengths = [len(l.split()) for l in test_lines]

    batch_lines = [[]]  # list of batch_lines
    prev_len = lengths[0]
    for length, line in zip(lengths, test_lines):
        # if same length as before,
        if prev_len == length:
            if len(batch_lines[-1]) < params.max_batch_size:
                batch_lines[-1].append(line)  # if previous batch still has some capacity, append to previous batch
            else:
                batch_lines.append([line])  # otherwise, create a new batch with same length
        # otherwise, create a new batch with this line
        else:
            batch_lines.append([line])
            prev_len = length

    return batch_lines


def generate_until(model, prefixes, stop_token, ml):
    """
    Generate a batch of string from a batch of prefixes
    :param model: pytorch model to sample from
    :param prefixes: list of strings (already BPEncoded)
    :param stop_token: stop generating at this token in each line
    :param ml: maximum length of tokens to generate
    :return: a list of generated string without the stop token and without the prefix
    """
    bs = len(prefixes)
    max_len = max([len(line.split()) for line in prefixes])

    for p in prefixes:
        if params.verbose: print(f'  [prefix.1] : {p}')
        assert len(p.split()) == max_len,\
            f"prefix length: {len(p.split())} -vs- max_len: {max_len}"  # assert all lines have the same length

    if params.verbose: print("  stop token: %s" % stop_token)

    with torch.no_grad():
        if torch.cuda.is_available():
            proc_prefixes = torch.LongTensor(bs, max_len).fill_(model.module.pad_index).cuda()  # bs x max_len
        else:
            proc_prefixes = torch.LongTensor(bs, max_len).fill_(model.module.pad_index)  # bs x max_len
        # fill in the tensor
        for idx, tokens in enumerate(prefixes):
            tokens = np.array([word2id.get(w.strip(), word2id['<unk>']) for w in tokens.split()])
            proc_prefixes[idx, :len(tokens)] = torch.from_numpy(tokens)
            # reconstruct the prefix with the appropriate vocab
            prefixes[idx] = ' '.join([id2word[w] for w in tokens])

        for p in prefixes:
            if params.verbose: print(f'  [prefix.2] : {p}')
        # prefix ~(bs, max_len)

        # generate
        decoded = model.module.generate(
            prefix=proc_prefixes, max_len=max_len + ml,
            sample=params.sample, temperature=params.temperature,
            stop_token=word2id[stop_token]
        )
        # decoded ~(max_len, bs)
        decoded = decoded.permute(1, 0)
        # decoded ~(bs, max_len)
        decoded = decoded.data.cpu().numpy().tolist()

    for i, decoded_line in enumerate(decoded):
        decoded_line = [id2word.get(x, '<unk>') for x in decoded_line]

        decoded_line = ' '.join(decoded_line).replace('\n', '')

        # ignore the prefix
        # --- NB ---
        # do this BEFORE cutting at the stop token!! Because otherwise,
        # if the stop token is part of the context (ex: '<STORY>'),
        # then we will ignore everything that was decoded
        # ----------
        decoded_line = decoded_line.replace(prefixes[i], '').strip()

        # cut the decoded line at the stop token
        # --- NB ---
        # do this AFTER removing the prefix!!
        # ----------
        # if generate until the end, take the first <ANSWER> and forget the rest
        if stop_token == '<pad>':
            decoded_line = decoded_line.split('<ANSWER>')
            if len(decoded_line) == 1:
                # if couldn't generate the end of answer ...
                print(f"couldn't generate the <ANSWER> token in '{decoded_line[0]}'")
                decoded[i] = re.split('<[A-Z]+>', decoded_line[0])[0].replace('\n', '') + '. . .'
                print(f"stop here: {decoded[i]}")
            else:
                # Now, also make sure we generated at least 1 sentence after the <ANSWER> tag.
                tmp = decoded_line[1].split('.')
                if len(tmp) == 1:
                    # splitting around '.' may not always work... sometimes the model doesn't generate a '.'
                    # so also split around TAGS
                    tmp = re.split('<[A-Z]+>', tmp[0])
                if len(tmp) == 1:
                    print(f"generated the <ANSWER> tag but no end-of-sentence in '{tmp[0]}'")
                    decoded[i] = decoded_line[0].replace('\n', '') + "<ANSWER>" + tmp[0].replace('\n', '') + ". . ."
                    print(f"stop here: {decoded[i]}")
                else:
                    # if we did generate the full sentence, then return this.
                    decoded[i] = decoded_line[0].replace('\n', '') + "<ANSWER>" + tmp[0].replace('\n', '') + '.'

        # if we do actually have a stop token, make sure we generated it and take everything before
        else:
            decoded_line = decoded_line.split(stop_token)
            if len(decoded_line) == 1:
                # if couldn't generate the stop token ...
                print(f"couldn't generate the '{stop_token}' token in '{decoded_line[0]}'")
                # ... or any other tag ...
                tmp = re.split('<[A-Z]+>', decoded_line[0])
                decoded[i] = tmp[0].replace('\n', '') + '. . .'
                print(f"stop here: {decoded[i]}")
            else:
                # if we did generate the full sentence, then
                # cut the generated sequence if a token was generated before the stop token
                decoded[i] = re.split('<[A-Z]+>', decoded_line[0])[0].replace('\n', '')

        # replace BPE tokens
        # decoded[i] = decoded[i].replace('@@ ', '')
        if params.verbose: print(f'  [decoded] : {decoded[i]}')

    return decoded


def load_model():
    print("Loading model...")
    model_args = pickle.load(
        open(os.path.join(params.load_path, 'args.pkl'), 'rb')
    )
    model = TransformerLM(
        n_words=len(word2id), emb_dim=model_args.emb_dim,
        n_layers=model_args.n_layers, dropout=model_args.dropout,
        pad_index=word2id['<pad>'],
        attention_heads=model_args.attention_heads,
        attention_dropout=model_args.attention_dropout,
        relu_dropout=model_args.relu_dropout,
        ffn_dim=model_args.transformer_ffn_emb_dim,
        freeze_emb=model_args.freeze_emb,
        freeze_limit=freeze_limit,
        activation=model_args.activation,
        adaptive_softmax=model_args.adaptive_softmax,
        adaptive_inputs=model_args.adaptive_inputs
    )

    model = nn.DataParallel(
        model, device_ids=list(range(torch.cuda.device_count()))
    )

    model_dict = torch.load(os.path.join(params.load_path, 'model.pt'))
    if "module.fixed_embeddings.weight" not in model_dict:
        model_dict["module.fixed_embeddings.weight"] = model_dict['module.embeddings.weight']
    model.load_state_dict(model_dict)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model has {num_params} parameters')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model has {num_params} trainable parameters')

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    return model


def main():
    model = load_model()

    # load existing file if any
    if os.path.isfile(params.out_file.replace('.txt', '.chkpnt')):
        print("Loading checkpoint...")
        with open(params.out_file.replace('.txt', '.chkpnt'), 'r') as f:
            meta = f.readlines()
        print("Loading done lines...")
        with open(params.out_file, 'r') as f:
            done_lines = f.readlines()
            # don't consider empty lines as done_lines!
            done_lines = np.array(list(filter(lambda l: len(l.strip()) > 0, done_lines)))
    else:
        done_lines = np.array([])
        meta = ["-1"]

    print("Loading test file...")
    with open(params.test_file, 'r') as f:
        all_test_lines = f.readlines()
    print(f"{len(all_test_lines)} lines loaded.")
    print(f"meta: {meta}")
    print(f"done lines: {len(done_lines)}")

    print("applying BPE and sorting lines by their lengths...")
    # apply BPE
    all_test_lines = bpe.apply(all_test_lines)
    # undo BPE on special tokens
    all_test_lines = undo_bpe(bpe, all_test_lines, fns_real)

    time_limit = time.time() + (params.save_frequency * 60 * 60)  # k hours = k*60*60

    # order the test lines by increasing length
    all_test_lines, idx_original_lengths = order_by_length(all_test_lines)
    total = len(all_test_lines)

    # build a list of batches. each batch containing a collection of same-length-strings
    all_test_lines = build_size_batches(all_test_lines)

    print("\n##################################################################################")
    print("\nGenerating proofs and answers...")

    processed = 1
    for bid, batch_lines in enumerate(all_test_lines):
        # ignore previously computed lines
        if bid <= int(meta[0]):
            processed += len(batch_lines)
            continue

        if params.verbose:
            print("\n---------------------------\n")
            for p in batch_lines:
                print(f'in: [{len(p.split())}] {p}')
        else:
            print(f"\nprocessing lines {processed} - {processed + len(batch_lines) - 1} / {total}. "
                  f"line lengths: {len(batch_lines[0].split())}")
        processed += len(batch_lines)

        prefixes = list(map(lambda line: line.split('<PROOF>')[0].strip() + ' <PROOF>', batch_lines))

        proofs_answers = generate_until(model, prefixes, '<pad>', params.max_length)

        for i, line in enumerate(batch_lines):
            done_lines = np.append(
                done_lines, line.replace('<PROOF> .@@ .@@ . <ANSWER> .@@ .@@ .', '<PROOF> ' + proofs_answers[i].strip())
            )

        # save after a while
        if time.time() > time_limit:
            print("save processed lines")
            # save processed lines
            with open(params.out_file, 'w') as f:
                f.writelines('\n'.join(done_lines))
            # save checkpoint
            with open(params.out_file.replace('.txt', '.chkpnt'), 'w') as f:
                f.writelines('\n'.join([str(bid)]))
            # reset timer
            time_limit = time.time() + (params.save_frequency * 60 * 60)

    print("\nput back all lines in their original order and save to file...")
    # put back in the original order
    all_test_lines = done_lines[idx_original_lengths]

    # undo BPE
    all_test_lines = list(map(lambda line: line.replace('@@ ', ''), all_test_lines))

    # save to file
    print(f"saving {len(all_test_lines)} lines to {params.out_file}...")
    with open(params.out_file, 'w') as f:
        f.writelines('\n'.join(all_test_lines))

    # delete checkpoint
    if os.path.isfile(params.out_file.replace('.txt', '.chkpnt')):
        os.remove(params.out_file.replace('.txt', '.chkpnt'))

    print("done.")


if __name__ == '__main__':
    params = get_args()
    params.verbose = (params.verbose == '1') or (params.verbose.lower() == 'yes') or (params.verbose.lower() == 'true')

    assert os.path.exists(params.load_path), f"{params.load_path} doesn't exists."
    assert os.path.exists(os.path.join(params.load_path, 'args.pkl')), f"{params.load_path}/args.pkl doesn't exists."
    assert os.path.exists(os.path.join(params.load_path, 'model.pt')), f"{params.load_path}/model.pt doesn't exists."
    assert os.path.exists(params.bpe_codes), f"{params.bpe_codes} doesn't exists."
    assert os.path.exists(params.gpt_vocab_path), f"{params.gpt_vocab_path} doesn't exists."
    assert os.path.exists(params.bpe_vocab_path), f"{params.bpe_vocab_path} doesn't exists."

    with open(params.gpt_vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    word2id, id2word, freeze_limit = vocab['word2id'], vocab['id2word'], vocab.get('freeze_limit', 0)

    tokenizer = MosesTokenizer(lang='en')
    detokenizer = MosesDetokenizer(lang='en')
    bpe = fastBPE.fastBPE(
        params.bpe_codes, params.bpe_vocab_path
    )

    # list of tokens to NOT split by BPE
    fns_real = [f"<hop{i}>" for i in range(1, 10+1)]  # <hop1> ... <hop10>
    fns_real.append("Since")
    fns_real += [f"ent_{d}" for d in range(1, 20+1)]  # ent_1 ... ent_20

    main()
