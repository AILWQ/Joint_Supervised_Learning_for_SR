import glob
import json
import marshal
import os
import pickle
import time
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import sympy
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from omegaconf import omegaconf
from tqdm import tqdm

from .dataset import generator
from .dclasses import DatasetDetails, Equation, GeneratorDetails


class H5FilesCreator():
    def __init__(self, base_path: Path = None, target_path: Path = None, metadata=None):
        target_path.mkdir(mode=0o777, parents=True, exist_ok=True)
        self.target_path = target_path
        self.base_path = base_path
        self.metadata = metadata

    def create_single_hd5_from_eqs(self, block):
        name_file, eqs = block
        t_hf = h5py.File(os.path.join(self.target_path, str(name_file) + ".h5"), 'w')
        for i, eq in enumerate(eqs):
            curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=curr)
        t_hf.close()

    def recreate_single_hd5_from_idx(self, block: Tuple):
        name_file, eq_idxs = block
        t_hf = h5py.File(os.path.join(self.target_path, str(name_file) + ".h5"), 'w')
        for i, eq_idx in enumerate(eq_idxs):
            eq = load_eq_raw(self.base_path, eq_idx, self.metadata.eqs_per_hdf)
            # curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=eq)
        t_hf.close()


def code_unpickler(data):
    return marshal.loads(data)


def code_pickler(code):
    return code_unpickler, (marshal.dumps(code),)


def load_eq_raw(path_folder, idx, num_eqs_per_set) -> Equation:
    index_file = str(int(idx / num_eqs_per_set))
    f = h5py.File(os.path.join(path_folder, f"{index_file}.h5"), 'r')
    dataset_metadata = f[str(idx - int(index_file) * int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    f.close()
    return raw_metadata


def load_eq(path_folder, idx, num_eqs_per_set) -> Equation:
    index_file = str(int(idx / num_eqs_per_set))
    f = h5py.File(os.path.join(path_folder, f"{index_file}.h5"), 'r')
    dataset_metadata = f[str(idx - int(index_file) * int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    f.close()
    return metadata


def load_metadata_hdf5(path_folder) -> DatasetDetails:
    f = h5py.File(os.path.join(path_folder, "metadata.h5"), 'r')
    dataset_metadata = f["other"]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    return metadata


def create_env(path) -> Tuple[generator.Generator, GeneratorDetails]:
    with open(path) as f:
        d = json.load(f)
    param = GeneratorDetails(**d)
    env = generator.Generator(param)
    return env, param, d


def generateDataFast(eq, n_points, n_vars, decimals, min_x, max_x, total_variabels=['x_1', 'x_2']):
    """
    :param eq: expression str with consts
    :param n_points: number of points
    :param min_x: min x
    :param max_x: max x
    :return:
    """
    X_ls = []
    Y_ls = []
    # X_dict = {}
    # temp = []
    start_time = time.time()
    p = 0
    while p < n_points:
        X_dict = {}
        temp = []
        X = np.round(np.random.uniform(min_x, max_x, n_vars), decimals=decimals)

        for x in total_variabels:
            if x in eq:
                X_dict.update({x: X[eval(x[-1]) - 1]})
                temp.append(x)
            else:
                X[-1] = 0.00000000
        # eq = sympy.sympify(eq)

        try:
            y = sympy.lambdify(",".join(temp), eq)(**X_dict)
        except:
            return [], []

        time_cost = time.time() - start_time
        if time_cost > 30.0:
            return [], []

        if np.isnan(y) or np.isinf(y):
            continue

        y = float(np.round(y, decimals=decimals))
        y = y if abs(y) < 5e4 else np.sign(y) * 5e4
        y = abs(y) if np.iscomplex(y) else y

        p += 1

        X_ls.append(list(X))
        Y_ls.append(y)

    # print(X_ls, '\n', shape(X_ls))
    # print(Y_ls, '\n', shape(Y_ls))
    return X_ls, Y_ls


def processDataFiles(files):
    text = ""
    for file in tqdm(files):
        with open(file, 'r') as f:
            lines = f.read()  # don't worry we won't run out of file handles
            if lines[-1] == -1:
                lines = lines[:-1]
            # text += lines #json.loads(line)
            text = ''.join([lines, text])
    return text


def filter_samples_not_in_word2id(text: list, word2id):
    remove_idx = []
    for i, example in enumerate(text):
        train_example = json.loads(example)
        traversal = train_example["traversal"]

        for idx in traversal:
            if idx not in list(word2id.keys()):
                remove_idx.append(i)
                break
            else:
                pass
    text = [e for i, e in enumerate(text) if i not in remove_idx]

    return text


def calculate_diff_vars_expressions():
    metadata_path = Path("")
    metadata = load_metadata_hdf5(metadata_path)
    cfg = omegaconf.OmegaConf.load("")

    # TODO: load the train dataset
    path = '{}/Train/*.json'.format('')

    files = glob.glob(path)

    text = processDataFiles(files)

    text = text.split('\n')  # convert the raw text to a set of examples

    trainText = text[:-1] if len(text[-1]) == 0 else text

    trainText = filter_samples_not_in_word2id(trainText, metadata.word2id)

    num_1var = 0
    num_2var = 0
    for chunk in trainText:
        dict = json.loads(chunk)
        expr = dict['eq']
        if 'x_2' in expr:
            num_2var += 1
            print('2vars：', num_2var)
        else:
            num_1var += 1
            print('1var：', num_1var)

    print('===> 1var：', num_1var)
    print('===> 2vars：', num_2var)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hours = int(elapsed_time / (60 * 60))
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_hours, elapsed_mins, elapsed_secs


def plot_losses(loss_file,
                plot_file,
                xlimits=None,
                ylimits=None):
    losses = [a.split() for a in open(loss_file).readlines()]
    train_losses = [float(z) for z, _ in losses]
    valid_losses = [float(z) for _, z in losses]

    epochs = np.arange(len(train_losses))

    # plot loss curves
    # set up an empty figure
    fig = plt.figure(figsize=(7, 4))

    # add a subplot to it
    nrows, ncols, index = 1, 1, 1
    ax = fig.add_subplot(nrows, ncols, index)

    ax.set_xlabel('epoch', fontsize=16)
    ax.set_ylabel('loss', fontsize=16)

    if xlimits: ax.set_xlim(xlimits)
    if ylimits: ax.set_ylim(ylimits)

    ax.plot(epochs, train_losses, c='red', label='training')
    ax.plot(epochs, valid_losses, c='blue', label='validation')

    ax.legend()

    plt.savefig(plot_file)
    plt.show()


def filter_samples_not_in_word2id(text: list, word2id):
    remove_idx = []
    for i, example in enumerate(text):
        train_example = json.loads(example)
        traversal = train_example["traversal"]

        for idx in traversal:
            if idx not in list(word2id.keys()):
                remove_idx.append(i)
                break
            else:
                pass
    text = [e for i, e in enumerate(text) if i not in remove_idx]

    return text


def MSE(y, y_pred):
    return torch.mean(torch.square(y - y_pred)).item()


def RMSE(y, y_pred):
    return torch.sqrt(torch.mean(torch.square(y - y_pred))).item()


def R_Square(y, y_pred):
    return (1 - torch.sum(torch.square(y - y_pred)) / torch.sum(torch.square(y - torch.mean(y)))).item()


def Relative_Error(y, y_pred):
    return torch.mean(torch.abs((y - y_pred) / y)).item()


def top_k_top_p_filtering(logits, top_k=0.0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).

        Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    # TODO: support for batch size more than 1 logits:[batchsize(1), seq_len, vocab_size]
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[0][sorted_indices_to_remove[0]]
        logits[..., indices_to_remove] = filter_value
    return logits
