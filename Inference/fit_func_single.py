from functools import partial
from pathlib import Path

import hydra
import omegaconf
import sympy
import torch
from sympy import lambdify

from src.architectures.expression_generator import model
from src.dclasses import FitParams, BFGSParams
from src.utils import R_Square, Relative_Error
from src.utils import generateDataFast
from src.utils import load_metadata_hdf5

# # specify the GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# load config
cfg = omegaconf.OmegaConf.load("../config.yaml")

metadata_path = Path("../data/raw_datasets/100000")  # load metadata
metadata = load_metadata_hdf5(hydra.utils.to_absolute_path(metadata_path))

# Set up BFGS load rom the hydra config yaml
bfgs = BFGSParams(
    activated=cfg.inference.bfgs.activated,
    n_restarts=cfg.inference.bfgs.n_restarts,
    add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
    normalization_o=cfg.inference.bfgs.normalization_o,
    idx_remove=cfg.inference.bfgs.idx_remove,
    normalization_type=cfg.inference.bfgs.normalization_type,
    stop_time=cfg.inference.bfgs.stop_time,
)

params_fit = FitParams(word2id=metadata.word2id,
                       # id2word={v: k for k,v in eq_setting["word2id"].items()},
                       id2word=metadata.id2word,
                       una_ops=metadata.una_ops,
                       bin_ops=metadata.bin_ops,
                       total_variables=list(metadata.total_variables),
                       total_coefficients=list(metadata.total_coefficients),
                       rewrite_functions=list(metadata.rewrite_functions),
                       bfgs=bfgs,
                       beam_size=cfg.inference.beam_size
                       # This parameter is a tradeoff between accuracy and fitting time)
                       )
# load model
weights_path = "../weight/*.pth"
model = model(cfg=cfg.architecture)
checkpoint = torch.load(weights_path)
model.load_state_dict(checkpoint['state_dict'])
if torch.cuda.is_available():
    model.cuda()
model.eval()

fitfunc = partial(model.fitfunc, cfg_params=params_fit)  # beam search
# fitfunc = partial(model.Nucleus_sampling, cfg_params=params_fit) # Nucleus_sampling

# Create points from an equation
number_of_points = 100
n_variables = 2

# To get best results make sure that your support inside the max and mix support
max_supp = cfg.dataset_train.fun_support["max"]  # 10
min_supp = cfg.dataset_train.fun_support["min"]  # -10

# ******************************************Benchmarks*******************************************#
# TODO:Nguyen
# target_eq = Nguyen_1 = "x_1**3 + x_1**2 + x_1"
# target_eq = Nguyen_2 = "x_1**4 + x_1**3 + x_1**2 + x_1"
# target_eq = Nguyen_3 = "x_1**5 + x_1**4 + x_1**3 + x_1**2 + x_1"
# target_eq = Nguyen_4 = "x_1**6 + x_1**5 + x_1**4 + x_1**3 + x_1**2 + x_1"
# target_eq = Nguyen_5 = "sin(x_1**2)*cos(x_1) - 1"
# target_eq = Nguyen_6 = "sin(x_1) + sin(x_1 + x_1**2)"
# target_eq = Nguyen_7 = "log(x_1+1)+log(x_1**2+1)"
# target_eq = Nguyen_8 = "x_1**0.5"
# target_eq = Nguyen_9 = "sin(x_1) + sin(x_2**2)"
# target_eq = Nguyen_10 = "2*sin(x_1)*cos(x_2)"
# target_eq = Nguyen_11 = "x_1**x_2"
# target_eq = Nguyen_12 = "x_1**4-x_1**3+0.5*x_2**2-x_2"

# TODO:Keijzer
# target_eq = Keijzer_3 =  "0.3 * x_1 * sin(2*pi*x_1)"
# target_eq = Keijzer_4 = "pow(x_1,3)*exp(-x_1)*cos(x_1)*sin(x_1)*(pow(sin(x_1),2)*cos(x_1)-1)"
# target_eq = Keijzer_9 = "log(x_1+sqrt(pow(x_1,2)+1))"
# target_eq = Keijzer_11 = "x_1*x_2+sin((x_1-1)*(x_2-1))"
# target_eq = Keijzer_13 = "6*sin(x_1)*cos(x_2)"
# target_eq = Keijzer_14 = "8/(2+x_1**2+x_2**2)"
# target_eq = Keijzer_15 = "0.2*x_1**3+0.5*x_2**3-x_2-x_1"

# TODO：Constant
# target_eq = Constant_1 = "3.39*pow(x_1,3)+2.12*pow(x_1,2)+1.78*x_1"
# target_eq = Constant_2 = "sin(pow(x_1,2))*cos(x_1)-0.75"
# target_eq = Constant_3 = "sin(1.5*x_1)*cos(0.5*x_2)"
# target_eq = Constant_4 = "2.7*pow(x_1,x_2)"
# target_eq = Constant_5 = "sqrt(1.23*x_1)"
# target_eq = Constant_6 = "pow(x_1,0.426)"
# target_eq = Constant_7 = "2*sin(1.3*x_1)*cos(x_2)"
# target_eq = Constant_8 = "log(x_1+1.4)+log(pow(x_1,2)+1.3)"

# R
# target_eq = R_1 = "(x_1+1)**3/(x_1**2-x_1+1)"
# target_eq = R_2 = "(x_1**5-3*x_1**3+1)/(x_1**2+1)"
# target_eq = R_3 = "(x_1**6+x_1**5)/(x_1**4+x_1**3+x_1**2+x_1+1)"

# TODO：Jin
# target_eq = Jin_1 = "2.5*x_1**4-1.3*x_1**3+0.5*x_2**2-1.7*x_2"

# TODO: AI Feynman
# target_eq = Feynman_1 = "exp(-x_1**2/2)/sqrt(2*pi)"
# target_eq = Feynman_2 = "exp(-(x_1/x_2)**2/2)/(sqrt(2*pi)*x_2)"
# target_eq = Feynman_3 = "x_1*x_2"
# target_eq = Feynman_4 = "x_1*x_2"
# target_eq = Feynman_5 = "0.5*x_1*x_2**2"
# target_eq = Feynman_6 = "x_1/x_2"
# target_eq = Feynman_7 = "1.5*x_1*x_2"
# target_eq = Feynman_8 = "x_1/(4*3.14*x_2**2)"
# target_eq = Feynman_9 = "(x_1*x_2**2)/2"
# target_eq = Feynman_10 = "x_1*x_2**2"
# target_eq = Feynman_11 = "x_1/(2*(1+x_2))"
# target_eq = Feynman_12 = "x_1*x_2/(2*3.14)"
# *****************************************************************************************#

# TODO: Custom
# target_eq = "x_1 + x_2"  # yse
# target_eq = "x_1*cos(tan(x_1))"
# target_eq = "2*(sin(x_1)+cos(x_1+4))"  # (3.51360498077642*tan(x_1) - 1.30728721001275)*cos(x_1)
# target_eq = "2*sin(x_1)+sin(x_1**2+x_1)+2"  # 4.67779908674875*(0.235426856590556*x_1 - 0.0873086803686236*exp(x_1) + 1)**9.3530600185381
target_eq = "x_1+exp(x_2*tan(x_1))"

# X = torch.rand(number_of_points, len(list(metadata.total_variables)))*(max_supp-min_supp)+min_supp
# X[:, n_variables:] = 0

# generate data points
X, y = generateDataFast(target_eq, n_points=100, n_vars=2, decimals=8, min_x=-10, max_x=10)
if len(y) == 0:
    print("y value is None !")
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X_dict = {x: X[:, idx].cpu() for idx, x in enumerate(metadata.total_variables)}
# y = lambdify(",".join(metadata.total_variables), target_eq)(**X_dict)

try:
    output = fitfunc(X, y)
    eq_pred = output['best_bfgs_preds'][0]
    y_pred = lambdify(",".join(metadata.total_variables), sympy.sympify(eq_pred))(**X_dict)
    print('***************** best preds *************************')
    print("expression: ", eq_pred)
    print("R^2: ", R_Square(y, y_pred))
    print("Relative_Error: ", Relative_Error(y, y_pred))
    print("MSE: ", torch.mean(torch.square(y - y_pred)).item())
    print('****************************************************')

except:
    print('Inference failed!')
