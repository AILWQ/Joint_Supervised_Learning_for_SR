import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

from baseline_utils import nostdout, processDataFiles
from src.utils import R_Square, Relative_Error

warnings.filterwarnings("ignore")

# GP configuration
POP_SIZE = 1000
GENERATIONS = 20
P_CROSSOVER = 0.9
WARM_START = False
const_range = (-5, 5)


def clipped_exp(x):
    return np.exp(np.clip(x, -999999, np.log(10000)))


def abs_sqrt(x):
    x = x.astype(float)
    return np.sqrt(np.abs(x))


def abs_log(x):
    x = x.astype(float)
    return np.log(np.sqrt(x * x + 1e-10))


exp_fn = make_function(clipped_exp, "exp", 1)
sqrt_fn = make_function(abs_sqrt, "sqrt", 1)
log_fn = make_function(abs_log, "log", 1)

FUNCTION_SET = ["add", "mul", "div", sqrt_fn, "sin", exp_fn, log_fn]


def generate_results(file_path, save_path):
    results_data = []
    test_set = processDataFiles([file_path])
    test_set = test_set.strip().split("\n")

    for idx, sample in enumerate(test_set):
        print("===> Inferencing No.{} ...".format(idx + 1))
        t = json.loads(sample)
        X = np.array(t["X"])
        y = np.array(t["y"])
        raw_eqn = t["eq"]

        try:
            start_time = time.time()
            with nostdout():
                model = SymbolicRegressor(
                    population_size=POP_SIZE,
                    generations=GENERATIONS, stopping_criteria=0,
                    p_crossover=P_CROSSOVER, p_subtree_mutation=0.1,
                    p_hoist_mutation=0.05, p_point_mutation=0.1,
                    warm_start=WARM_START,
                    max_samples=0.9, verbose=False,
                    parsimony_coefficient=0.001,
                    function_set=FUNCTION_SET,
                    n_jobs=-1,
                    const_range=const_range
                )
                model.fit(X, y)
                equation_pred = model._program
                y_pred = model.predict(X)

            train_time = time.time() - start_time
            y, y_pred = torch.from_numpy(y), torch.from_numpy(y_pred)
            MSE = torch.mean(torch.square(y - y_pred)).item()
            R2 = R_Square(y, y_pred)
            RE = Relative_Error(y, y_pred)
            # predicted_tree = model._program
            print("===> Inference success！\n")
        except ValueError:
            print("===> Inference failed！\n")
            equation_pred = "NA"
            predicted_tree = "NA"
            MSE = "NA"
            R2 = "NA"
            RE = "NA"
            train_time = "NA"

        results_data.append({
            "test_index": idx,
            "true_equation": raw_eqn,
            # "true_skeleton": skeleton,
            "predicted_equation": equation_pred,
            # "predicted_tree": predicted_tree,
            # "predicted_y": y_pred,
            "MSE": MSE,
            "R2": R2,
            "RE": RE,
            "inference_time": train_time
        })

    results = pd.DataFrame(results_data)
    results.to_csv(save_path, index=False)


if __name__ == "__main__":
    data_path = "../Dataset/2_var/1000000/SSDNC/*.json"
    save_path = "./results/SSDNC_gp.csv"

    generate_results(data_path, save_path)
