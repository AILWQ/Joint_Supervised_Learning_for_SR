import torch
from transformers import top_k_top_p_filtering

from src.architectures import bfgs
from src.dataset.generator import InvalidPrefixExpression


def Nucleus_sampling(self, X, y, cfg_params=None):
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
        points_embeddings = points_embeddings.unsqueeze(1)

        generated = torch.zeros([1, self.cfg.length_eq], dtype=torch.long, device=self.device)

        generated[:, 0] = 1

        cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)
        while cur_len < self.cfg.length_eq:

            token_embeddings = self.tok_emb(generated[:,
                                            :cur_len])  # [2, seq_len-1, emb_dim]  # each index maps to a (learnable) vector -> (batchsize x length x embedding)
            position_embeddings = self.pos_emb(torch.arange(0, cur_len)
                                               .unsqueeze(0)
                                               .repeat(generated.shape[0], 1)
                                               .type_as(generated))

            input_embedding = token_embeddings + position_embeddings + points_embeddings  # [2, blocksize, embeddingsize]
            x = self.drop(input_embedding)  # [batchsize, blocksize, embeddingsize]
            x = self.blocks(x)
            x = self.ln_f(x)  # + points_embeddings:[batch, length, embedding]
            logits = self.head(x)  # [batchsize, seq_length, vocab_size]

            logits = top_k_top_p_filtering(logits[:, -1, :], top_k=0, top_p=0.7)

            probs = F.softmax(logits, dim=-1)

            token = torch.multinomial(probs, num_samples=1)

            generated[:, cur_len] = token

            if (token == cfg_params.word2id["F"] or cur_len + 1 == self.cfg.length_eq):
                break

            # update current length
            cur_len = cur_len + torch.tensor(1, device=self.device, dtype=torch.int64)

        # BFGS optimize constants
        L_bfgs = []
        P_bfgs = []
        generated = generated[0]

        try:
            pred_w_c, constants, loss_bfgs, exa = bfgs.bfgs(generated[:cur_len], X, y, cfg_params)
        except InvalidPrefixExpression:
            print('Invalid Prefix Expression!')

        P_bfgs.append(str(pred_w_c))
        L_bfgs.append(loss_bfgs)

        output = {'best_bfgs_preds': P_bfgs, 'all_bfgs_loss': L_bfgs}

        return output
