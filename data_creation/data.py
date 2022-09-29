import math
import warnings
from typing import List

import torch
from sympy import sympify, Float, Symbol
from sympy.core.rules import Transform
from torch.distributions.uniform import Uniform

from src.dclasses import Equation


def custom_collate_fn(eqs: List[Equation], cfg) -> List[torch.tensor]:
    filtered_eqs = [eq for eq in eqs if eq.valid]
    res, tokens = evaluate_and_wrap(filtered_eqs, cfg)
    return res, tokens, [eq.expr for eq in filtered_eqs]


def constants_to_placeholder(s, symbol="c"):
    sympy_expr = sympify(s)  # self.infix_to_sympy("(" + s + ")")
    sympy_expr = sympy_expr.xreplace(
        Transform(
            lambda x: Symbol(symbol, real=True, nonzero=True),
            lambda x: isinstance(x, Float),
        )
    )
    return sympy_expr


def tokenize(prefix_expr: list, word2id: dict) -> list:
    tokenized_expr = []
    tokenized_expr.append(word2id["S"])
    for i in prefix_expr:
        tokenized_expr.append(word2id[i])
    tokenized_expr.append(word2id["F"])
    return tokenized_expr


def tokenize_with_no_S_F(prefix_expr: list, word2id: dict) -> list:
    tokenized_expr = []
    for i in prefix_expr:
        tokenized_expr.append(word2id[i])
    return tokenized_expr


def de_tokenize(tokenized_expr, id2word: dict):
    prefix_expr = []
    for i in tokenized_expr:
        if "F" == id2word[i]:
            break
        else:
            prefix_expr.append(id2word[i])
    return prefix_expr


def tokens_padding(tokens):
    max_len = max([len(y) for y in tokens])
    p_tokens = torch.zeros(len(tokens), max_len)
    for i, y in enumerate(tokens):
        y = torch.tensor(y).long()
        p_tokens[i, :] = torch.cat([y, torch.zeros(max_len - y.shape[0]).long()])
    return p_tokens


def number_of_support_points(p, type_of_sampling_points):
    if type_of_sampling_points == "constant":
        curr_p = p
    elif type_of_sampling_points == "logarithm":
        curr_p = int(10 ** Uniform(1, math.log10(p)).sample())
    else:
        raise NameError
    return curr_p


def sample_support(eq, curr_p, cfg):
    sym = []
    if not eq.support:
        distribution = torch.distributions.Uniform(cfg.fun_support.min,
                                                   cfg.fun_support.max)  # torch.Uniform.distribution_support(cfg.fun_support[0],cfg.fun_support[1])
    else:
        raise NotImplementedError

    for sy in cfg.total_variables:
        if sy in eq.variables:
            curr = distribution.sample([int(curr_p)])
        else:
            curr = torch.zeros(int(curr_p))
        sym.append(curr)
    return torch.stack(sym)


def sample_constants(eq, curr_p, cfg):
    consts = []
    # eq_c = set(eq.coeff_dict.values())
    for c in cfg.total_coefficients:
        if c[:2] == "cm":
            if c in eq.coeff_dict:
                curr = torch.ones([int(curr_p)]) * eq.coeff_dict[c]
            else:
                curr = torch.ones([int(curr_p)])
        elif c[:2] == "ca":
            if c in eq.coeff_dict:
                curr = torch.ones([int(curr_p)]) * eq.coeff_dict[c]
            else:
                curr = torch.zeros([int(curr_p)])
        consts.append(curr)
    return torch.stack(consts)


def evaluate_and_wrap(eqs: List[Equation], cfg):
    vals = []
    cond0 = []
    tokens_eqs = [eq.tokenized for eq in eqs]
    tokens_eqs = tokens_padding(tokens_eqs)
    curr_p = number_of_support_points(cfg.max_number_of_points, cfg.type_of_sampling_points)
    for eq in eqs:
        support = sample_support(eq, curr_p, cfg)
        consts = sample_constants(eq, curr_p, cfg)
        input_lambdi = torch.cat([support, consts], axis=0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aaaa = eq.code(*input_lambdi)
                if type(aaaa) == torch.Tensor and aaaa.dtype == torch.float32:
                    vals.append(
                        torch.cat(
                            [support, torch.unsqueeze(aaaa, axis=0)], axis=0
                        ).unsqueeze(0)
                    )
                    cond0.append(True)
                else:
                    cond0.append(False)
        except NameError as e:
            # print(e)
            cond0.append(False)
        except RuntimeError as e:
            cond0.append(False)
        # except:
        #     breakpoint()
    tokens_eqs = tokens_eqs[cond0]
    num_tensors = torch.cat(vals, axis=0)
    cond = (
            torch.sum(torch.count_nonzero(torch.isnan(num_tensors), dim=2), dim=1)
            < curr_p / 25
    )
    num_fil_nan = num_tensors[cond]
    tokens_eqs = tokens_eqs[cond]
    cond2 = (
            torch.sum(
                torch.count_nonzero(torch.abs(num_fil_nan) > 5e4, dim=2), dim=1
            )  # Luca comment 0n 21/01
            < curr_p / 25
    )
    num_fil_nan_big = num_fil_nan[cond2]
    tokens_eqs = tokens_eqs[cond2]
    idx = torch.argsort(num_fil_nan_big[:, -1, :]).unsqueeze(1).repeat(1, num_fil_nan_big.shape[1], 1)
    res = torch.gather(num_fil_nan_big, 2, idx)
    # res, _ = torch.sort(num_fil_nan_big)
    res = res[:, :, torch.sum(torch.count_nonzero(torch.isnan(res), dim=1), dim=0) == 0]
    res = res[
          :,
          :,
          torch.sum(torch.count_nonzero(torch.abs(res) > 5e4, dim=1), dim=0)
          == 0,  # Luca comment 0n 21/01
          ]
    return res, tokens_eqs
