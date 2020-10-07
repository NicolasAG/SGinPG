"""Data iterator things."""
from logging import getLogger
import os
import numpy as np
import torch
import operator
import pickle
from collections import OrderedDict
logger = getLogger()


class DataIterator(object):
    """Data Iterator."""

    def _trim_vocab(self, vocab, vocab_size):
        # Discard start, end, pad and unk tokens if already present
        if '<s>' in vocab:
            del vocab['<s>']
        if '<pad>' in vocab:
            del vocab['<pad>']
        if '</s>' in vocab:
            del vocab['</s>']
        if '<unk>' in vocab:
            del vocab['<unk>']
        if '<blank>' in vocab:
            del vocab['<blank>']
        if '<cls>' in vocab:
            del vocab['<cls>']
        if '<sep>' in vocab:
            del vocab['<sep>']
        if '<seg1>' in vocab:
            del vocab['<seg1>']
        if '<seg2>' in vocab:
            del vocab['<seg2>']

        next_idx = 0

        word2id = {
            '<pad>': next_idx,
            '<s>': next_idx + 1,
            '</s>': next_idx + 2,
            '<unk>': next_idx + 3,
            '<blank>': next_idx + 4,
            '<cls>': next_idx + 5,
            '<sep>': next_idx + 6,
            '<seg1>': next_idx + 7,
            '<seg2>': next_idx + 8
        }

        id2word = {
            next_idx: '<pad>',
            next_idx + 1: '<s>',
            next_idx + 2: '</s>',
            next_idx + 3: '<unk>',
            next_idx + 4: '<blank>',
            next_idx + 5: '<cls>',
            next_idx + 6: '<sep>',
            next_idx + 7: '<seg1>',
            next_idx + 8: '<seg2>'
        }

        # Sort vocab based on frequency
        sorted_word2id = sorted(
            vocab.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # Pick only vocab_size number of items if != -1
        if vocab_size != -1:
            sorted_words = [x[0] for x in sorted_word2id[:vocab_size]]
        else:
            sorted_words = [x[0] for x in sorted_word2id]

        for ind, word in enumerate(sorted_words):
            word2id[word] = ind + next_idx + 9
            id2word[ind + next_idx + 9] = word

        assert len(word2id) == len(id2word), f"word2id: {len(word2id)} != id2word: {len(id2word)}"

        return word2id, id2word, next_idx

    def construct_vocab(self, sentences, vocab_size, lowercase=False):
        """Create vocabulary."""
        vocab = {}
        for sentence in sentences:
            if isinstance(sentence, str):
                if lowercase:
                    sentence = sentence.lower()
                sentence = sentence.split()
            for word in sentence:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
        assert np.all(['\x00' not in w for w in vocab])
        word2id, id2word, freeze_limit = self._trim_vocab(vocab, vocab_size)
        assert np.all(['\x00' not in w for w in word2id])
        return word2id, id2word, freeze_limit


class MultiBPTTAutoIterator(object):
    """Automatically get next minibatch"""
    def __init__(self, info_dict, cache_dir, cache):
        self.multi_iterator = MultiBPTTIterator(info_dict, cache_dir, cache)
        # keep track of where we are in a buffer for each dataset
        self.indices = {k: 0 for k in info_dict.keys()}

    def __len__(self):
        """Number of datasets."""
        return len(self.multi_iterator)

    def fetch_buffer(self, key):
        """Fetch lines into the buffer for a dataset."""
        self.multi_iterator.fetch_buffer(key)

    def get_minibatch(self, key, batch_size):
        """Get next minibatch for a dataset."""
        is_last = False
        batch = self.multi_iterator.get_minibatch(key, self.indices[key], batch_size)
        self.indices[key] += batch_size
        if self.indices[key] >= self.multi_iterator.iterators[key].buffer_tensor.size(0):
            self.fetch_buffer(key)
            self.indices[key] = 0
            is_last = True
        return batch, is_last


class MultiBPTTIterator(object):
    """A wrapper around BPTTIterator to include multiple datasets [WIP]."""

    def __init__(self, info_dict, cache_dir, cache):
        """Initialize iterators."""
        self.iterators = OrderedDict()
        for k, v in info_dict.items():
            self.iterators[k] = BPTTDataIterator(
                fname=v['fname'], vocab_size=v['vocab_size'],
                bptt_length=v['bptt_length'], cache_dir=v['cache_dir'],
                cache=v['cache'], buffer_size=v['buffer_size'],
                skip_vocab_saving=True
            )

        # different folder for the merged vocab
        self.cache_dir = cache_dir
        self.cache = cache

        self._merge_vocabs()

    def __len__(self):
        """Number of datasets."""
        return len(self.iterators)

    def _merge_vocabs(self):
        if not os.path.exists(
            os.path.join(self.cache_dir, 'vocab.pkl')
        ):
            logger.warning('Could not find cached MERGED vocab at %s, recomputing...' %(
                os.path.join(self.cache_dir, 'vocab.pkl')
            ))
            # couldn't find a cached vocab file, set to False to build one
            self.cache = False
        if os.path.exists(
            os.path.join(self.cache_dir, 'vocab.pkl')
        ) and self.cache:
            logger.info('Reloading cached MERGED vocab file at : %s' % (
                os.path.join(self.cache_dir, 'vocab.pkl')
            ))
            vocab = pickle.load(
                open(os.path.join(self.cache_dir, 'vocab.pkl'), 'rb')
            )
            self.word2id, self.id2word = vocab['word2id'], vocab['id2word']
            self.freeze_limit = vocab.get('freeze_limit', 0)

        if not self.cache:
            logger.info('Building MERGED vocabulary into %s...' % (
                os.path.join(self.cache_dir, 'vocab.pkl')
            ))

            # Get the list of unique words
            words = list(set(sum([
                list(itt.id2word.values()) for _, itt in self.iterators.items()], []
            )))
            assert np.all(['\x00' not in w for w in words])
            # merge dictionaries
            self.word2id, self.id2word, self.freeze_limit = self._proper_dicts(words)
            assert np.all(['\x00' not in w for w in self.word2id])
            vocab = {'word2id': self.word2id, 'id2word': self.id2word, 'freeze_limit': self.freeze_limit}
            pickle.dump(
                vocab, open(os.path.join(self.cache_dir, 'vocab.pkl'), 'wb')
            )

        for _, itrt in self.iterators.items():
            itrt.word2id = self.word2id
            itrt.id2word = self.id2word
            itrt.freeze_limit = self.freeze_limit

        print(f"vocab size: {len(self.word2id)}")
        assert set(range(len(self.word2id))) == set(self.word2id.values()), \
            f"\n[w2id] range - ids = {set(range(len(self.word2id))) - set(self.word2id.values())}" \
            f"\n[w2id] ids - range = {set(self.word2id.values()) - set(range(len(self.word2id)))}"
        assert set(range(len(self.word2id))) == set(self.id2word.keys()), \
            f"\n[id2w] range - ids = {set(range(len(self.word2id))) - set(self.id2word.keys())}" \
            f"\n[id2w] ids - range = {set(self.id2word.keys()) - set(range(len(self.word2id)))}"

    def _proper_dicts(self, words):
        word2id = {word: idx for idx, word in enumerate(words)}
        id2word = {idx: word for idx, word in enumerate(words)}
        return word2id, id2word, 0

    def reset_fptr(self, key):
        """Reset the file pointer for a dataset."""
        self.iterators[key]._reset_fptr()

    def fetch_buffer(self, key):
        """Fetch lines into the buffer for a dataset."""
        self.iterators[key].fetch_buffer()

    def get_minibatch(self, key, idx, batch_size):
        """Get minibatch for a dataset."""
        return self.iterators[key].get_minibatch(idx, batch_size)


class BPTTDataIterator(DataIterator):
    """Iterator for Language modeling and Masked Language Modeling (BERT)."""

    def __init__(
            self, fname, vocab_size, bptt_length, cache_dir, cache, buffer_size=-1,
            skip_vocab_saving=False
    ):
        """Initialize params."""
        self.fname = fname
        self.vocab_size = vocab_size
        self.bptt_length = bptt_length
        self.cache_dir = cache_dir
        self.cache = cache
        self.buffer_size = buffer_size
        assert os.path.exists(cache_dir)
        self._reset_fptr()
        self.compute_vocab(skip_vocab_saving)
        self._reset_fptr()

    def _reset_fptr(self):
        self.fptr = open(self.fname, 'r')

    def compute_vocab(self, skip_vocab_saving):
        """
        Compute vocab.
        :param skip_vocab_saving: save the vocab.pkl file or not
        """
        if not os.path.exists(
            os.path.join(self.cache_dir, 'vocab.pkl')
        ):
            logger.warning('Could not find cached vocab at %s, recomputing...' % (
                os.path.join(self.cache_dir, 'vocab.pkl')
            ))
            # couldn't find a cached vocab file, set to False to build one
            self.cache = False
        if os.path.exists(
            os.path.join(self.cache_dir, 'vocab.pkl')
        ) and self.cache:
            logger.info('Reloading cached vocab file at : %s' % (
                os.path.join(self.cache_dir, 'vocab.pkl')
            ))
            vocab = pickle.load(
                open(os.path.join(self.cache_dir, 'vocab.pkl'), 'rb')
            )
            self.word2id, self.id2word = vocab['word2id'], vocab['id2word']
            self.freeze_limit = vocab.get('freeze_limit', 0)  # if word_id < freeze_limit: pick freeze embedding

        if not self.cache:
            if not skip_vocab_saving:
                logger.info('Building vocabulary into %s ...' % (os.path.join(self.cache_dir, 'vocab.pkl')))
            else:
                logger.info('Building vocabulary ...')
            self.word2id, self.id2word, self.freeze_limit = self.construct_vocab(
                self.fptr, self.vocab_size
            )
            # if word_id < freeze_limit: pick freeze embedding
            if not skip_vocab_saving:
                vocab = {'word2id': self.word2id, 'id2word': self.id2word, 'freeze_limit': self.freeze_limit}
                pickle.dump(
                    vocab, open(os.path.join(self.cache_dir, 'vocab.pkl'), 'wb')
                )

    def fetch_buffer(self):
        """Fetch examples to fill the buffer."""
        if self.buffer_size == -1:
            self._reset_fptr()
        print(f'Buffer empty, fetching examples from {self.fname}...')

        assert len(self.id2word) == len(self.word2id)
        print(f"vocab size: {len(self.word2id)}")
        assert set(range(len(self.word2id))) == set(self.word2id.values()), \
            f"\n[w2id] range - ids = {set(range(len(self.word2id))) - set(self.word2id.values())}" \
            f"\n[w2id] ids - range = {set(self.word2id.values()) - set(range(len(self.word2id)))}"
        assert set(range(len(self.word2id))) == set(self.id2word.keys()), \
            f"\n[id2w] range - ids = {set(range(len(self.word2id))) - set(self.id2word.keys())}" \
            f"\n[id2w] ids - range = {set(self.id2word.keys()) - set(range(len(self.word2id)))}"

        buffer_tensor = []
        # nwords = 0  # essentially just the length of cur_batch
        cur_batch = []
        sentences_processed = 0
        truncated_sentences = 0
        completed_sentences = 0
        for idx, sentence in enumerate(self.fptr):
            if idx == self.buffer_size and self.buffer_size != -1:
                break
            for word in sentence.split():
                # if nwords == self.bptt_length:
                if len(cur_batch) == self.bptt_length:
                    # print(f"\nreached end of batch ({len(cur_batch)}) with this:")
                    # print(' '.join([self.id2word[w] for w in cur_batch]))
                    buffer_tensor.append(cur_batch)
                    cur_batch = []
                    # nwords = 0
                    truncated_sentences += 1
                # start a new current batch for each story
                if len(cur_batch) > 0 and word == '<STORY>':
                    # print(f"\nstop this batch here ({len(cur_batch)}):")
                    remaining_before_end = self.bptt_length - len(cur_batch)
                    cur_batch.extend([self.word2id['<pad>']] * remaining_before_end)
                    # print(f"stop this batch here ({len(cur_batch)}):")
                    # print(' '.join([self.id2word[w] for w in cur_batch]))
                    buffer_tensor.append(cur_batch)
                    cur_batch = []
                    # nwords = 0
                    completed_sentences += 1
                cur_batch.append(
                    self.word2id[word] if word in self.word2id
                    else self.word2id['<unk>']
                )
                # nwords += 1
            sentences_processed += 1

        # If the end of the dataset is reached, reset fptr and fetch again
        if sentences_processed != self.buffer_size and self.buffer_size != -1:
            print('Reaching end of dataset, fetching again')
            self._reset_fptr()
            self.fetch_buffer()

        print(f"processed sentences: {sentences_processed}")
        print(f"total examples = {completed_sentences + truncated_sentences} = completed ({completed_sentences}) + truncated ({truncated_sentences})")

        buffer_tensor = torch.LongTensor(buffer_tensor)
        random_idxs = torch.randperm(buffer_tensor.size(0))
        self.buffer_tensor = buffer_tensor[random_idxs]

    def get_minibatch(self, idx, batch_size):
        """Return a minibatch of input -> output."""
        data = self.buffer_tensor[idx: idx + batch_size]
        return data[:, :-1], data[:, 1:]
