"""Transformer LMs from Radford et al (2018/2019).

Adapted from fairseq-py/Unsupervised MT/XLM
"""
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    """Gaussian Error Linear Unit."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


class AdaptiveInput(nn.Module):
    """Adaptive input representations from Baevski et al. (2019)."""

    def __init__(
        self, vocab_size, padding_idx, initial_dim, factor, output_dim, cutoff
    ):
        """Init params."""
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert vocab_size == cutoff[
                -1], 'cannot specify cutoff larger than vocab size'

        self.cutoff = cutoff
        self.embedding_dim = output_dim
        self.padding_idx = padding_idx

        self.embeddings = nn.ModuleList()
        for i in range(len(self.cutoff)):
            prev = self.cutoff[i - 1] if i > 0 else 0
            size = self.cutoff[i] - prev
            dim = int(initial_dim // (factor ** i))

            seq = nn.Sequential(
                nn.Embedding(size, dim, padding_idx),
                nn.Linear(dim, output_dim, bias=False)
            )
            self.embeddings.append(seq)

        def init_weights(m):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(
                    m.weight, mean=0, std=m.weight.shape[1] ** -0.5
                )
                nn.init.constant_(m.weight[padding_idx], 0)
            elif hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def weights_for_band(self, band):
        """Util for tying weights to adaptive softmax."""
        return self.embeddings[band][0].weight, self.embeddings[band][1].weight

    def forward(self, input):
        """Forward through the layer."""
        result = self._float_tensor.new(input.shape + (self.embedding_dim,))
        for i in range(len(self.cutoff)):
            mask = input.lt(self.cutoff[i])
            if i > 0:
                mask.mul_(input.ge(self.cutoff[i - 1]))
                chunk_input = input[mask] - self.cutoff[i - 1]
            else:
                chunk_input = input[mask]
            if mask.any():
                result[mask] = self.embeddings[i](chunk_input)
        return result


class MultiheadAttention(nn.Module):
    """Multi-headed attention."""

    def __init__(self, embed_dim, num_heads, dropout=0.):
        """Init params."""
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            self.head_dim * num_heads, self.embed_dim
        )
        self.scaling = self.head_dim ** -0.5
        self._mask = None

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, query, key, value, mask_future_timesteps=True,
        key_padding_mask=None, need_weights=True
    ):
        """Forward through the layer."""
        """Note: This only implements Self-attention where query == key == value
        """
        assert query.data_ptr() == key.data_ptr() == value.data_ptr()

        seq_len, bsz, embed_dim = query.size()

        assert embed_dim == self.embed_dim
        assert list(query.size()) == [seq_len, bsz, embed_dim]
        assert key.size() == value.size()

        # self-attention
        q, k, v = self.in_proj(query).chunk(3, dim=-1)
        q *= self.scaling

        assert key_padding_mask.size(0) == bsz, (
            key_padding_mask.size(), bsz, key.size(), query.size()
        )
        assert key_padding_mask.size(1) == seq_len, (
            key_padding_mask.size(), seq_len
        )

        q = q.contiguous().view(
            seq_len, bsz * self.num_heads, self.head_dim
        ).transpose(0, 1)  # ~(bs*n_heads , seq_len , head_dim)
        k = k.contiguous().view(
            seq_len, bsz * self.num_heads, self.head_dim
        ).transpose(0, 1)  # ~(bs*n_heads , seq_len , head_dim)
        v = v.contiguous().view(
            seq_len, bsz * self.num_heads, self.head_dim
        ).transpose(0, 1)  # ~(bs*n_heads , seq_len , head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # ~(bs*n_heads , seq_len , seq_len)
        assert list(attn_weights.size()) == [
            bsz * self.num_heads, seq_len, seq_len
        ]

        # Mask future timesteps
        if mask_future_timesteps:
            attn_weights += self.buffered_mask(
                attn_weights.data
            ).detach().unsqueeze(0)

        if key_padding_mask.data.max() > 0:
            attn_weights = attn_weights.view(
                bsz, self.num_heads, seq_len, seq_len
            )

            # Mask attention weights corresponding to <pad> tokens
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                -float('inf'),
            )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, seq_len, seq_len
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [
            bsz * self.num_heads, seq_len, self.head_dim
        ]
        attn = attn.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(
            bsz, self.num_heads, seq_len, seq_len
        )
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def buffered_mask(self, tensor):
        """Masking for attention.
        torch.triu() returns the upper right triangle of a 2D matrix,
          the rest of the values are set to 0.
        By filling in the 2D matrix with -inf we will ignore those attention weights
          other attention weights won't be impacted (value 0).
        diagonal=1 means that the diagonal attention weights are also set to 0
          (ie: you can attend to yourself).
        """
        dim = tensor.size(-1)
        if self._mask is None:
            self._mask = torch.triu(
                tensor.new(dim, dim).fill_(-float('inf')), diagonal=1
            )
        if self._mask.size(0) < dim:
            self._mask = torch.triu(
                self._mask.resize_(dim, dim).fill_(-float('inf')), diagonal=1
            )
        return self._mask[:dim, :dim]


def create_sinusoidal_embeddings(n_pos, dim, out):
    """Sinusoidal position embeddings."""
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    # Make the position embeddings not learnable.
    out.detach_()
    out.requires_grad = False


class TransformerLM(nn.Module):
    """Transformer encoder."""

    def __init__(
        self, n_words, emb_dim, n_layers, dropout, pad_index, attention_heads,
        attention_dropout, relu_dropout, ffn_dim, freeze_emb=False, freeze_limit=0,
        activation='relu', adaptive_softmax=False, asm_cutoffs=[8000, 20000],
        asm_div_value=4, adaptive_inputs=False
    ):
        """Init encoder params."""
        super().__init__()
        self.dropout = dropout
        self.n_words = n_words
        self.pad_index = pad_index
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.freeze_emb = freeze_emb
        self.freeze_limit = freeze_limit
        self.adaptive_softmax = adaptive_softmax
        self.adaptive_inputs = adaptive_inputs

        if self.adaptive_inputs:
            self.embeddings = AdaptiveInput(
                n_words, pad_index, emb_dim,
                asm_div_value, emb_dim, asm_cutoffs
            )
        else:
            self.embeddings = nn.Embedding(
                n_words, emb_dim, padding_idx=pad_index
            )
            nn.init.normal_(
                self.embeddings.weight, mean=0, std=emb_dim ** -0.5
            )
            nn.init.constant_(self.embeddings.weight[pad_index], 0)

        # Define a set of fixed embeddings as well
        self.fixed_embeddings = nn.Embedding(n_words, emb_dim, padding_idx=pad_index)
        nn.init.normal_(
            self.fixed_embeddings.weight, mean=0, std=emb_dim ** -0.5
        )
        self.fixed_embeddings.weight.requires_grad = False

        self.embed_scale = emb_dim ** 0.5

        self.embed_positions = nn.Embedding(2048, self.emb_dim)
        self.activation = F.relu if activation == 'relu' else gelu
        create_sinusoidal_embeddings(
            2048, self.emb_dim, out=self.embed_positions.weight
        )

        layers = []
        for k in range(n_layers):
            layers.append(TransformerLMLayer(
                emb_dim=emb_dim, dropout=dropout, num_heads=attention_heads,
                attention_dropout=attention_dropout, relu_dropout=relu_dropout,
                ffn_dim=ffn_dim, activation=self.activation
            ))
        self.layers = nn.ModuleList(layers)

        if self.adaptive_softmax:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=emb_dim,
                n_classes=n_words,
                cutoffs=asm_cutoffs,
                div_value=asm_div_value,
                head_bias=False,
            )
        else:
            self.proj = nn.Linear(emb_dim, n_words)

    def forward(self, input, output, mask_future_timesteps=True):
        """Forward through the layer.
        input  ~(bs, seq)
        output ~(bs, seq)
        """
        hidden = self._get_hidden(input, mask_future_timesteps)  # ~(seq, bs, emb_dim)
        hidden = hidden.transpose(1, 0)                          # ~(bs, seq, emb_dim)
        # Transpose output to be batch first again for DataParallel gather.

        if self.adaptive_softmax:
            _, loss = self.proj(
                hidden.contiguous().view(-1, hidden.size(2)),  # ~(bs * seq, emb_dim)
                output.contiguous().view(-1)                   # ~(bs * seq)
            )
            return loss

        else:
            logits = self.proj(hidden)  # ~(bs, seq, n_words)
            return self.loss_fn(
                logits.contiguous().view(-1, logits.size(2)),  # ~(bs * seq, n_words)
                output.contiguous().view(-1)                   # ~(bs * seq)
            )

    def _get_logits(self, input, mask_future_timesteps):
        hidden = self._get_hidden(input, mask_future_timesteps)  # ~(seq, bs, emb_dim)
        if not self.adaptive_softmax:
            return self.proj(hidden)  # ~(seq, bs, n_words)
        else:
            hidden_flat = hidden.view(-1, hidden.size(2))
            logits = self.proj.log_prob(hidden_flat)
            return logits.view(
                hidden.size(0), hidden.size(1), logits.size(1)
            )

    def _get_hidden(self, input, mask_future_timesteps):
        """
        input ~(bs, seq)
        """
        # Make things sequence first because DataParallel requires batch first.
        src_tokens = input.t()  # ~(seq, bs)

        # Create position locations.
        positions = src_tokens.new(src_tokens.size(0)).long()
        positions = torch.arange(
            src_tokens.size(0), out=positions
        ).unsqueeze(1)  # ~(seq,)

        ''' DEBUG ' ' '
        print(f"source tokens:")
        print(src_tokens)
        print(f"embeddings:")
        print(self.embeddings)
        print("frozen embeddings:")
        print(self.fixed_embeddings)
        '''

        # Embedding lookup
        x = self.embed_scale * self.embeddings(src_tokens)  # ~(seq, bs, emb_dim)
        x = x.detach() if self.freeze_emb else x

        # Construct a partially frozen embedding
        # from https://discuss.pytorch.org/t/partially-freeze-embedding-layer/18458/6
        if self.freeze_limit > 0:
            frozen_x = self.embed_scale * self.fixed_embeddings(src_tokens)  # ~(seq, bs, emb_dim)
            frozen_x = frozen_x.detach()  # to be sure it is frozen
            final_x = []
            for i, words in enumerate(src_tokens.data):
                freeze_emb = frozen_x[i]  # ~(bs, emb_dim)
                regular_emb = x[i]        # ~(bs, emb_dim)
                seq_emb = []
                for j, w_id in enumerate(words):
                    if w_id < self.freeze_limit:
                        seq_emb.append(freeze_emb[j])
                    else:
                        seq_emb.append(regular_emb[j])
                seq_emb = torch.stack(seq_emb)  # ~(bs, emb_dim)
                final_x.append(seq_emb)
            final_x = torch.stack(final_x)  # ~(seq, bs, emb_dim)
        else:
            final_x = x

        ''' DEBUG ' ' '
        print("positional embeddings:")
        print(self.embed_positions)
        print("positions_BEFORE")
        print(positions)
        print("positions_AFTER")
        print(positions.expand_as(src_tokens))
        '''

        # Add position information
        final_x = final_x + self.embed_positions(positions.expand_as(src_tokens))  # ~(seq, bs, emb_dim)
        final_x = F.dropout(final_x, p=self.dropout, training=self.training)       # ~(seq, bs, emb_dim)

        # compute padding mask
        padding_mask = src_tokens.t().eq(self.pad_index)  # ~(bs, seq)

        # encoder layers
        for layer in self.layers:
            final_x = layer(final_x, padding_mask, mask_future_timesteps)
        return final_x  # ~(seq, bs, emb_dim)

    def generate(
        self, prefix=None, max_len=512, sample='topk',
        temperature=0.7, k=30, nucleus=0.9, stop_token=None
    ):
        """Generate outputs given an initial state."""
        """Prefix:
            - LongTensor of size (batch_size, seq_len) representing sentences
        Output:
            - LongTensor of size (max_len, batch_size), word indices
        """
        if prefix is None:
            assert temperature is not None
            assert sample

        assert sample in ['nucleus', 'topk', 'temperature', 'greedy']
        # initialize generated sentences batch
        cur_len = 1
        bs = prefix.size(0) if prefix is not None else 32
        decoded = torch.LongTensor(max_len, bs).fill_(self.pad_index)
        if prefix is not None:
            prefix = prefix.t()
            decoded[:prefix.size(0), :prefix.size(1)] = prefix
            cur_len = prefix.size(0)
        if torch.cuda.is_available():
            decoded = decoded.cuda()

        while cur_len < max_len:
            # previous word embeddings
            scores = self._get_logits(
                decoded[:cur_len].transpose(1, 0), mask_future_timesteps=True
            )
            scores = scores.data[-1, :, :]  # T x B x V -> B x V

            if sample == 'topk':
                assert k > 0
                # Get the top k scores
                topk, _ = torch.topk(scores, k, -1)
                topk = topk[:, -1].unsqueeze(1)
                # Mask out all values < lowest of the topk
                if scores.dtype == torch.float16:
                    mask = torch.gt(scores, topk).half()
                else:
                    mask = torch.gt(scores, topk).float()
                scores = scores * mask
                probs = F.softmax(scores / temperature, -1)
                next_words = torch.multinomial(probs, 1).squeeze(1)
            elif sample == 'nucleus':
                probs = F.softmax(scores / temperature, -1)
                # Sort in descending order to do cumsum
                sorted_probs, idx = torch.sort(probs, dim=-1, descending=True)
                # Define cumultative prob cut-off
                cutoff_mask = sorted_probs.cumsum(-1) < nucleus
                # Atleast the most likely element should be included
                cutoff_mask[:, 0] = 1
                # Mask out low prob tokens
                if scores.dtype == torch.float16:
                    sorted_probs = sorted_probs * cutoff_mask.half()
                else:
                    sorted_probs = sorted_probs * cutoff_mask.float()
                # Sample word
                sampled_idx = torch.multinomial(sorted_probs, 1).squeeze(1)
                # Go back to original index (before sorting)
                next_words = idx[torch.arange(idx.size(0)).cuda(), sampled_idx]
            elif sample == 'temperature':
                probs = F.softmax(scores / temperature, -1)
                next_words = torch.multinomial(probs, 1).squeeze(1)
            elif sample == 'greedy':
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            assert next_words.size() == (bs,)

            '''
            print('decoded[cur_len-1]: %s' % decoded[cur_len-1])
            print('next words: %s' % next_words)
            print('stop word: %s' % stop_token)
            if stop_token:
                # check previous word,
                for b_idx, prev_w in enumerate(decoded[cur_len-1]):
                    # if it was a stop word,
                    if prev_w == stop_token:
                        print("previous word was stop token!")
                        # ignore prediction and repeat the stop word again
                        decoded[cur_len, b_idx] = stop_token
            else:
            '''
            decoded[cur_len] = next_words

            if stop_token and (decoded[cur_len] == stop_token).all():
                break
            else:
                cur_len += 1

        return decoded


class TransformerLMLayer(nn.Module):
    """Transformer Encoder layer block."""

    """A single residual transformer block.
    Operations:
    self attn -> dropout -> add residual -> layernorm -> feed forward -> \
        add residual -> layernorm
    """
    def __init__(
        self, emb_dim, dropout, num_heads,
        attention_dropout, relu_dropout, ffn_dim, activation
    ):
        """Init params."""
        super().__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ffn_dim = ffn_dim
        self.self_attn = MultiheadAttention(
            self.emb_dim, self.num_heads,
            dropout=self.attention_dropout,
        )
        self.relu_dropout = relu_dropout
        self.fc1 = nn.Linear(self.emb_dim, self.ffn_dim)
        self.fc2 = nn.Linear(self.ffn_dim, self.emb_dim)
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.emb_dim) for i in range(2)]
        )
        self.activation = activation

    def forward(self, x, padding_mask, mask_future_timesteps):
        """Forward through the layer."""
        residual = x
        x, _ = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=padding_mask,
            mask_future_timesteps=mask_future_timesteps
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if x.dtype == torch.float16:
            x = x.float()
        x = self.layer_norms[0](x)
        if residual.dtype == torch.float16:
            x = x.half()

        residual = x
        x = self.activation(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.activation(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if x.dtype == torch.float16:
            x = x.float()
        x = self.layer_norms[1](x)
        if residual.dtype == torch.float16:
            x = x.half()
        return x
