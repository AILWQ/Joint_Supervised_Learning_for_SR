import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from feature_extractor import pointMLP
from src.architectures import bfgs
from src.architectures.beam_search import BeamHypotheses
from src.dataset.generator import InvalidPrefixExpression


class CausalSelfAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        assert cfg.dim_hidden % cfg.num_heads == 0

        # TODO: key, query, value projections for all heads
        self.key = nn.Linear(cfg.dim_hidden, cfg.dim_hidden)
        self.query = nn.Linear(cfg.dim_hidden, cfg.dim_hidden)
        self.value = nn.Linear(cfg.dim_hidden, cfg.dim_hidden)

        # TODO: regularization
        self.attn_drop = nn.Dropout(cfg.attn_drop)
        self.resid_drop = nn.Dropout(cfg.resid_drop)

        # output projection
        self.proj = nn.Linear(cfg.dim_hidden, cfg.dim_hidden)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(cfg.length_eq, cfg.length_eq))
                             .view(1, 1, cfg.length_eq, cfg.length_eq))
        self.num_heads = cfg.num_heads

    def forward(self, x, layer_past=None):
        # x: [batchsize, blocksize, embeddingsize]

        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1,
                                                                                  2)  # (batchsize, nheads, length, emb//nheads)
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # # [batchsize, nheads, blocksize, blocksize]
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  # [batchsize, nheads, blocksize, blocksize]
        att = self.attn_drop(att)  # [batchsize, nheads, blocksize, blocksize]
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> [batchsize, nheads, blocksize, emb//nheads]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [batchsize, blocksize, embeddingsize]

        # output projection
        y = self.resid_drop(self.proj(y))  # [batchsize, blocksize, embeddingsize]
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim_hidden)
        self.ln2 = nn.LayerNorm(cfg.dim_hidden)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim_hidden, 4 * cfg.dim_hidden),
            nn.GELU(),
            nn.Linear(4 * cfg.dim_hidden, cfg.dim_hidden),
            nn.Dropout(cfg.resid_drop),
        )

    def forward(self, x):
        # x: [batchsize, blocksize, embeddingsize]

        x = x + self.attn(self.ln1(x))  # [batchsize, blocksize, embeddingsize]
        # x = x + self.multiheadattention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))  # [batchsize, blocksize, embeddingsize]
        return x


class model(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.pointMLP = pointMLP()

        self.block_size = cfg.length_eq

        # TODO: input embedding stem
        self.tok_emb = nn.Embedding(cfg.output_dim, cfg.dim_hidden, padding_idx=cfg.src_pad_idx)
        self.pos_emb = nn.Embedding(cfg.length_eq, cfg.dim_hidden)
        self.drop = nn.Dropout(cfg.embd_pdrop)

        # TODO: transformer block
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.dec_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(cfg.dim_hidden)

        self.head = nn.Linear(cfg.dim_hidden, cfg.output_dim, bias=False)

        self.apply(self._init_weights)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Parameter)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('affine_alpha'):
                    decay.add(fpn)
                elif pn.endswith('affine_beta'):
                    decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr, betas=train_config.betas)
        return optimizer

    def forward(self, points=None, trg=None):
        '''
        points: [batchsize, num_X+y, num_points]
        trg: [batchsize, seq_len]
        '''

        global loss_features
        trg = trg.long()
        # TODO: forward the GPT model
        token_embeddings = self.tok_emb(trg[:,
                                        :-1])  # [batch_size, seq_len-1, emb_dim]  # each index maps to a (learnable) vector -> (batchsize x length x embedding)
        position_embeddings = self.pos_emb(torch.arange(0, trg.shape[1] - 1)
                                           .unsqueeze(0)
                                           .repeat(trg.shape[0], 1)
                                           .type_as(trg))  # [batch_size, seq_len-1, emb_dim]

        if points != None:

            points_embeddings = self.pointMLP(points)  # [batch_size, emb_dim]

            loss_features = points_embeddings  # [batch_size, fea_dim]

            # TODO:points_embedding:[batchsize, embeddingsize]->[batchsize, 1, embeddingsize]->[batchisize, blocksize/length, embeddingsize]
            points_embeddings = points_embeddings.unsqueeze(1)  # [batchisize, 1, embeddingsize]

            points_embeddings = torch.tile(points_embeddings, (
                1, token_embeddings.shape[1], 1))  # [batchisize, blocksize/length, embeddingsize]

            input_embedding = token_embeddings + position_embeddings + points_embeddings  # [batchsize, blocksize, embeddingsize]

        else:
            input_embedding = token_embeddings + position_embeddings  # [batchsize, blocksize, embeddingsize]

        x = self.drop(input_embedding)  # [batchsize, blocksize, embeddingsize]
        x = self.blocks(x)
        x = self.ln_f(x)  # + points_embeddings:[batch, length, embedding]

        logits = self.head(x)  # [batchsize, length, vocab_size]
        logits = logits.permute(1, 0, 2)

        return logits, trg, loss_features

    def fitfunc(self, X, y, cfg_params=None):
        """Same API as fit functions in sklearn:
            X [Number_of_points, Number_of_features],
            Y [Number_of_points]
        """
        X = X
        y = y[:, None]  # [num_points, 1]

        X = torch.tensor(X, device=self.device).unsqueeze(0)  # [1, num_points, num_features]
        if X.shape[2] < self.cfg.dim_input - 1:
            pad = torch.zeros(1, X.shape[1], self.cfg.dim_input - X.shape[2] - 1, device=self.device)
            X = torch.cat((X, pad), dim=2)
        y = torch.tensor(y, device=self.device).unsqueeze(0)  # [1, num_points, 1]

        with torch.no_grad():

            encoder_input = torch.cat((X, y), dim=2)  # .permute(0, 2, 1) [1, num_points, num_X+y]

            encoder_input = encoder_input.permute(0, 2, 1)  # [1, num_X+y, num_points]

            points_embeddings = self.pointMLP(encoder_input)
            points_embeddings = points_embeddings.unsqueeze(1).repeat(cfg_params.beam_size, 1,
                                                                      1)  # [2, blocksize/length, embeddingsize]

            generated = torch.zeros([cfg_params.beam_size, self.cfg.length_eq], dtype=torch.long, device=self.device)

            generated[:, 0] = 1

            cache = {"slen": 0}
            # generated = torch.tensor(trg_indexes,device=self.device,dtype=torch.long)

            generated_hyps = BeamHypotheses(cfg_params.beam_size, self.cfg.length_eq, 1.0, 1)

            done = False

            # Beam Scores
            beam_scores = torch.zeros(cfg_params.beam_size, device=self.device, dtype=torch.long)
            beam_scores[1:] = -1e9
            # beam_scores = beam_scores.view(-1)

            cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)  # 当前表达式长度为1，<BOS>
            while cur_len < self.cfg.length_eq:

                token_embeddings = self.tok_emb(generated[:,
                                                :cur_len])  # [2, seq_len-1, emb_dim]  # each index maps to a (learnable) vector -> (batchsize x length x embedding)
                position_embeddings = self.pos_emb(torch.arange(0, cur_len)
                                                   .unsqueeze(0)
                                                   .repeat(generated.shape[0], 1)
                                                   .type_as(generated))

                # input_embedding = token_embeddings + position_embeddings + points_embeddings  # [2, blocksize, embeddingsize]
                input_embedding = token_embeddings + position_embeddings + points_embeddings[:, :cur_len,
                                                                           :]  # merge encoder
                x = self.drop(input_embedding)  # [batchsize, blocksize, embeddingsize]
                x = self.blocks(x)
                x = self.ln_f(x)  # + points_embeddings:[batch, length, embedding]
                logits = self.head(x)  # [batchsize, seq_length, vocab_size]
                output = logits.permute(1, 0, 2)

                output = output.permute(1, 0, 2).contiguous()  # [2, seq_len, vocab_size]

                scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(1)

                assert output[:, -1:, :].shape == (cfg_params.beam_size, 1, self.cfg.length_eq,)

                n_words = scores.shape[-1]
                # select next words with scores
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
                _scores = _scores.view(cfg_params.beam_size * n_words)  # (bs, beam_size * n_words)

                next_scores, next_words = torch.topk(_scores, 2 * cfg_params.beam_size, dim=0, largest=True,
                                                     sorted=True)
                assert len(next_scores) == len(next_words) == 2 * cfg_params.beam_size
                done = done or generated_hyps.is_done(next_scores.max().item())
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words, next_scores):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if (
                            word_id == cfg_params.word2id["F"]
                            or cur_len + 1 == self.cfg.length_eq
                    ):
                        generated_hyps.add(
                            generated[beam_id, :cur_len, ]
                            .clone()
                            .cpu(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append(
                            (value, word_id, beam_id)
                        )

                    # the beam for next step is full
                    if len(next_sent_beam) == cfg_params.beam_size:
                        break

                # update next beam content
                assert (
                    len(next_sent_beam) == 0
                    if cur_len + 1 == self.cfg.length_eq
                    else cfg_params.beam_size
                )
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                                         (0, self.cfg.trg_pad_idx, 0)
                                     ] * cfg_params.beam_size  # pad the batch

                # next_batch_beam.extend(next_sent_beam)
                assert len(next_sent_beam) == cfg_params.beam_size

                beam_scores = torch.tensor(
                    [x[0] for x in next_sent_beam], device=self.device
                )  # .type(torch.int64) Maybe #beam_scores.new_tensor([x[0] for x in next_batch_beam])
                beam_words = torch.tensor(
                    [x[1] for x in next_sent_beam], device=self.device
                )  # generated.new([x[1] for x in next_batch_beam])
                beam_idx = torch.tensor(
                    [x[2] for x in next_sent_beam], device=self.device
                )
                generated = generated[beam_idx, :]
                generated[:, cur_len] = beam_words
                for k in cache.keys():
                    if k != "slen":
                        cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

                # update current length
                cur_len = cur_len + torch.tensor(
                    1, device=self.device, dtype=torch.int64
                )

            best_preds_bfgs = []
            best_L_bfgs = []

            L_bfgs = []
            P_bfgs = []

            cfg_params.id2word[3] = "constant"
            for __, ww in sorted(generated_hyps.hyp, key=lambda x: x[0], reverse=True):
                try:
                    pred_w_c, constants, loss, exa = bfgs.bfgs(ww, X, y, cfg_params)

                except InvalidPrefixExpression:
                    continue
                P_bfgs.append(str(pred_w_c))
                L_bfgs.append(loss)

            if all(np.isnan(np.array(L_bfgs))):
                print("Warning all nans")
                L_bfgs = float("nan")
                best_L_bfgs = None
            else:
                best_preds_bfgs.append(P_bfgs[np.nanargmin(L_bfgs)])
                best_L_bfgs.append(np.nanmin(L_bfgs))

            output = {'all_bfgs_preds': P_bfgs, 'all_bfgs_loss': L_bfgs, 'best_bfgs_preds': best_preds_bfgs,
                      'best_bfgs_loss': best_L_bfgs}
            self.eq = output['best_bfgs_preds']
            return output

    def encoder_feature(self, X, y):
        """
            return latent features by encoder
        """

        X = X
        y = y[:, None]  # [num_points, 1]

        X = torch.tensor(X, device=self.device).unsqueeze(0)  # [1, num_points, num_features]
        if X.shape[2] < self.cfg.dim_input - 1:
            pad = torch.zeros(1, X.shape[1], self.cfg.dim_input - X.shape[2] - 1, device=self.device)
            X = torch.cat((X, pad), dim=2)
        y = torch.tensor(y, device=self.device).unsqueeze(0)  # [1, num_points, 1]

        with torch.no_grad():
            encoder_input = torch.cat((X, y), dim=2)  # .permute(0, 2, 1) [1, num_points, num_X+y]
            encoder_input = encoder_input.permute(0, 2, 1)  # [1, num_X+y, num_points]
            points_embeddings = self.pointMLP(encoder_input)  # [1, dim_hid]
        return points_embeddings
