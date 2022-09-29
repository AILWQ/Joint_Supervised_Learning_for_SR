import warnings

import numpy as np
import torch
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

from src.utils import generateDataFast
from src.utils import R_Square, Relative_Error

warnings.filterwarnings("ignore")

# GP configuration
POP_SIZE = 1000
GENERATIONS = 20
P_CROSSOVER = 0.9
WARM_START = False
const_range = (-5, 5)

# Data Configuration
min_x = -10
max_x = 10
n_points = 100
n_vars = 2


def clipped_exp(x):
    return np.exp(np.clip(x, -999999, np.log(10000)))


def abs_sqrt(x):
    x = x.astype(float)
    return np.sqrt(np.abs(x))


def abs_log(x):
    x = x.astype(float)
    return np.log(np.sqrt(x * x + 1e-10))


exp_fn = make_function(function=clipped_exp, name="exp", arity=1)
sqrt_fn = make_function(function=abs_sqrt, name="sqrt", arity=1)
log_fn = make_function(function=abs_log, name="log", arity=1)

FUNCTION_SET = ["add", "mul", "div", "sin", "cos", "log", "sqrt"]

# ******************************************Public Benchmark*******************************************#
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
# target_eq = Keijzer_3 = "0.3 * x_1 * sin(2*pi*x_1)"
# target_eq = Keijzer_4 = "pow(x_1,3)*exp(-x_1)*cos(x_1)*sin(x_1)*(pow(sin(x_1),2)*cos(x_1)-1)"
# target_eq = Keijzer_6 = "(x_1*(x_1+1))/2"
# target_eq = Keijzer_7 = "log(x_1)"
# target_eq = Keijzer_8 = "sqrt(x_1)"
# target_eq = Keijzer_9 = "log(x_1+sqrt(pow(x_1,2)+1))"
# target_eq = Keijzer_11 = "x_1*x_2+sin((x_1-1)*(x_2-1))"
# target_eq = Keijzer_12 = "x_1**4-x_1**3+0.5*x_2**2-x_2"
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
target_eq = Feynman_12 = "x_1*x_2/(2*3.14)"
# *****************************************************************************************#

X, y = generateDataFast(target_eq, n_points=n_points, n_vars=n_vars, decimals=8, min_x=min_x, max_x=max_x)
if len(y) == 0:
    print("y is None !")
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

model = SymbolicRegressor(
    population_size=POP_SIZE,
    generations=GENERATIONS,
    p_crossover=P_CROSSOVER,
    warm_start=WARM_START,
    verbose=1,
    parsimony_coefficient=0.001,
    function_set=FUNCTION_SET,
    n_jobs=-1,
    const_range=const_range)

model.fit(X, y)
equation_pred = model._program
y_pred = model.predict(X)

MSE = torch.mean(torch.square(y - y_pred)).item()
R2 = R_Square(y, y_pred)
RE = Relative_Error(y, y_pred)
print("predict: ", equation_pred)
print("R^2: ", R2)
print("RE: ", RE)
