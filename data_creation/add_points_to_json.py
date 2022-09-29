import json
import os
import re
import warnings
from datetime import datetime

import hydra
import numpy as np
import omegaconf
import sympy
from data import constants_to_placeholder
from src.dataset import data_utils
from src.dataset.generator import Generator, UnknownSymPyOperator
from src.utils import generateDataFast, load_eq, load_metadata_hdf5
from tqdm import tqdm

warnings.filterwarnings("ignore")

cfg = omegaconf.OmegaConf.load("../config.yaml")

dataset_path = '../data/raw_datasets/100000'
metadata = load_metadata_hdf5(hydra.utils.to_absolute_path(dataset_path))

n_vars = 2
n_points = 100
number_per_equation = 50

fileID = 0
now = datetime.now()
file_other = '{}Points_'.format(n_points) + now.strftime("%d%m%Y_%H%M%S")
folder = '../Dataset/2_var/{}/Train'.format(number_per_equation*100000)
dataPath = folder + '/{}_{}Vars_{}.json'

################Train set###############
count = 0
i = 0
pbar = tqdm(total=metadata.total_number_of_eqs * number_per_equation)
while i < metadata.total_number_of_eqs:
    structure = {}
    eq = load_eq(dataset_path, i, metadata.eqs_per_hdf)

    eq_no_consts = eq.expr_infix
    consts_elemns = eq.coeff_dict

    j = 0
    while j < number_per_equation:
        break_out_flag = False
        w_const, wout_consts = data_utils.sample_symbolic_constants_from_coeff_dict(consts_elemns,
                                                                                    cfg.dataset_train.constants)
        eq_string = eq_no_consts.format(**w_const)
        eq_infix = str(sympy.sympify(eq_string)).replace(' ', '')
        if 'zoo' in eq_infix or 'nan' in eq_infix or "I" in eq_infix or "E" in eq_infix or "pi" in eq_infix:
            j += 1
            continue

        # modify exponent
        exps = re.findall(r"(\*\*[0-9\.]+)", eq_infix)
        for ex in exps:
            # correct the exponent
            cexp = '**' + str(eval(ex[2:]) if eval(ex[2:]) < 6 else np.random.randint(2, 6))
            # replace the exponent
            eq_infix = eq_infix.replace(ex, cexp)

        try:
            eq_sympy_infix = constants_to_placeholder(eq_infix)
            eq_skeleton = str(eq_sympy_infix).replace(' ', '')
            traversal = Generator.sympy_to_prefix(eq_sympy_infix)
        except UnknownSymPyOperator as e:
            print(e)
            j += 1
            continue
        except RecursionError as e:
            print(e)
            j += 1
            continue

        for val in traversal:
            if val not in list(metadata.word2id.keys()):
                break_out_flag = True
                break
        if break_out_flag:
            print(traversal)
            j += 1
            continue

        # generate data points
        points = generateDataFast(eq_infix, n_points=n_points, n_vars=n_vars, decimals=8, min_x=-10, max_x=10)
        X, y = points

        if len(y) == 0:
            j += 1
            continue

        structure["X"] = X  # Val/Test
        structure["y"] = y  # Val/Test
        structure["eq"] = eq_infix  # Val/Test
        structure["skeleton"] = eq_skeleton
        structure["traversal"] = traversal

        outputPath = dataPath.format(fileID, n_vars, file_other)
        if os.path.exists(outputPath):
            fileSize = os.path.getsize(outputPath)
            if fileSize > 300000000:
                fileID += 1
        with open(outputPath, "a", encoding="utf-8") as h:
            json.dump(structure, h, ensure_ascii=False)
            h.write('\n')
        j += 1
        count += 1
        pbar.update(1)

    i += 1

pbar.close()
h.close()
print("Total non-complex expressions: {}".format(count))

########## Val/Test##########
# count = 0
# i = 0
# pbar = tqdm(total=1000) # Val/Test
# while i < 1000: # Val/Test
#     break_out_flag = False
#     structure = {}
#     id = np.random.randint(0, 100000)
#     eq = load_eq(dataset_path, id, metadata.eqs_per_hdf)
#
#     eq_no_consts = eq.expr_infix
#     consts_elemns = eq.coeff_dict
#
#     w_const, wout_consts = data_utils.sample_symbolic_constants_from_coeff_dict(consts_elemns, cfg.dataset_train.constants)
#     eq_string = eq_no_consts.format(**w_const)
#     eq_infix = str(sympy.sympify(eq_string)).replace(' ', '')
#     if 'zoo' in eq_infix or 'nan' in eq_infix or "I" in eq_infix or "E" in eq_infix or "pi" in eq_infix:
#         i += 1
#         continue
#
#     exps = re.findall(r"(\*\*[0-9\.]+)", eq_infix)
#     for ex in exps:
#         # correct the exponent
#         cexp = '**' + str(eval(ex[2:]) if eval(ex[2:]) < 6 else np.random.randint(2, 6))
#         # replace the exponent
#         eq_infix = eq_infix.replace(ex, cexp)
#
#     try:
#         eq_sympy_infix = constants_to_placeholder(eq_infix)
#         eq_skeleton = str(eq_sympy_infix).replace(' ', '')
#         traversal = Generator.sympy_to_prefix(eq_sympy_infix)
#     except UnknownSymPyOperator as e:
#         print(e)
#         i += 1
#         continue
#     except RecursionError as e:
#         print(e)
#         i += 1
#         continue
#
#     for val in traversal:
#         if val not in list(metadata.word2id.keys()):
#             break_out_flag = True
#             break
#     if break_out_flag:
#         print('filtered traversal: ', traversal)
#         i += 1
#         continue
#
#     points = generateDataFast(eq_infix, n_points=n_points, n_vars=n_vars, decimals=8, min_x=-10, max_x=10)
#     X, y = points
#
#     if len(y) == 0:
#         i += 1
#         continue
#
#     structure["X"] = X # Val/Test
#     structure["y"] = y # Val/Test
#     structure["eq"] = eq_infix # Val/Test
#     structure["skeleton"] = eq_skeleton
#     structure["traversal"] = traversal
#
#     outputPath = dataPath.format(fileID, n_vars, file_other)
#     if os.path.exists(outputPath):
#         fileSize = os.path.getsize(outputPath)
#         if fileSize > 250000000:  # 100 MB
#             fileID += 1
#     with open(outputPath, "a", encoding="utf-8") as h:
#         json.dump(structure, h, ensure_ascii=False)
#         h.write('\n')
#
#     count += 1
#     pbar.update(1)
#     i += 1
#
# pbar.close()
# h.close()
# print("Total good expressions: {}".format(count))
