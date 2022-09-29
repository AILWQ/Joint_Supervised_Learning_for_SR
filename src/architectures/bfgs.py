import time

import numpy as np
import sympy as sp
import torch
from scipy.optimize import minimize

from src.dataset.generator import Generator


class TimedFun:
    def __init__(self, fun, stop_after=10):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(*x, *args)
        self.x = x
        return self.fun_value


def de_tokenize(tokenized_expr, id2word: dict):
    prefix_expr = []
    for i in tokenized_expr:
        if "F" == id2word[i]:
            break
        else:
            prefix_expr.append(id2word[i])
    return prefix_expr


def bfgs(pred_str, X, y, cfg):
    # Check where dimensions not use, and replace them with 1 to avoid numerical issues with BFGS (i.e. absent variables placed in the denominator)
    y = y.squeeze()
    X = X.clone()
    bool_dim = (X == 0).all(axis=1).squeeze()
    X[:, :, bool_dim] = 1

    pred_str = pred_str[1:].tolist()
    raw = de_tokenize(pred_str, cfg.id2word)  # 先序列表

    candidate = Generator.prefix_to_infix(raw,
                                          coefficients=["constant"],
                                          variables=cfg.total_variables)
    candidate = candidate.format(constant="constant")

    expr = candidate
    for i in range(candidate.count("constant")):
        expr = expr.replace("constant", f"c{i}", 1)

    print('Constructing BFGS loss...')

    if cfg.bfgs.idx_remove:
        print('Flag idx remove ON, Removing indeces with high values...')
        bool_con = (X < 200).all(axis=2).squeeze()
        X = X[:, bool_con, :]

    max_y = np.max(np.abs(torch.abs(y).cpu().numpy()))
    print('checking input values range...')
    if max_y > 300:
        print('Attention, input values are very large. Optimization may fail due to numerical issues')

    diffs = []
    for i in range(X.shape[1]):
        curr_expr = expr
        for idx, j in enumerate(cfg.total_variables):
            curr_expr = sp.sympify(curr_expr).subs(j, X[:, i, idx])
        diff = curr_expr - y[i]
        diffs.append(diff)

    if cfg.bfgs.normalization_o:
        raise NotImplementedError

    if cfg.bfgs.normalization_type == "NMSE":  # and (mean != 0):
        mean_y = np.mean(y.numpy())
        if abs(mean_y) < 1e-06:
            print("Normalizing by a small value")
        loss = (np.mean(np.square(diffs))) / mean_y
    elif cfg.bfgs.normalization_type == "MSE":
        loss = (np.mean(np.square(diffs)))
    else:
        raise KeyError

    print('Loss constructed, starting new BFGS optmization...')

    # Lists where all restarted will be appended
    F_loss = []
    consts_ = []
    funcs = []
    symbols = {i: sp.Symbol(f'c{i}') for i in range(candidate.count("constant"))}

    for i in range(cfg.bfgs.n_restarts):
        # Compute number of coefficients
        x0 = np.random.randn(len(symbols))
        s = list(symbols.values())
        # bfgs optimization
        fun_timed = TimedFun(fun=sp.lambdify(s, loss, modules=['numpy']), stop_after=cfg.bfgs.stop_time)
        if len(x0):
            minimize(fun_timed.fun, x0, method='BFGS')  # check consts interval and if they are int
            consts_.append(fun_timed.x)
        else:
            consts_.append([])

        final = expr
        for i in range(len(s)):
            final = sp.sympify(final).replace(s[i], fun_timed.x[i])
        if cfg.bfgs.normalization_o:
            funcs.append(max_y * final)
        else:
            funcs.append(final)

        values = {x: X[:, :, idx].cpu() for idx, x in enumerate(cfg.total_variables)}  # CHECK ME
        y_found = sp.lambdify(",".join(cfg.total_variables), final)(**values)
        final_loss = np.mean(np.square(y_found - y.cpu()).numpy())
        F_loss.append(final_loss)

    try:
        k_best = np.nanargmin(F_loss)
    except ValueError:
        print("All-Nan slice encountered")
        k_best = 0
    return funcs[k_best], consts_[k_best], F_loss[k_best], expr
