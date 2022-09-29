import copyreg
import multiprocessing
import os
import pickle
import time
import traceback
import types
import warnings
from itertools import chain
from pathlib import Path

import click
import h5py
import numpy as np
import sympy
from tqdm import tqdm

from data import constants_to_placeholder
from src import dclasses
from src.dataset import generator, data_utils
from src.dataset.generator import Generator, UnknownSymPyOperator
from src.utils import code_unpickler, code_pickler
from src.utils import create_env, H5FilesCreator


class Pipepile:

    def __init__(self, env: generator.Generator, number_of_expressions, eq_per_block, h5_creator: H5FilesCreator,
                 is_timer=False, cfg=None):
        self.env = env
        self.is_timer = is_timer
        self.number_of_expressions = number_of_expressions
        self.fun_args = ",".join(chain(list(env.variables), env.coefficients))
        self.eq_per_block = eq_per_block
        self.h5_creator = h5_creator
        self.cfg = cfg

    def create_block(self, block_idx):
        block = []
        counter = block_idx
        hlimit = block_idx + self.eq_per_block

        while counter < hlimit and counter < self.number_of_expressions:
            res = self.return_training_set(counter)
            block.append(res)
            counter = counter + 1
        self.h5_creator.create_single_hd5_from_eqs((block_idx // self.eq_per_block, block))
        return 1

    def handler(self, signum, frame):
        raise TimeoutError

    def return_training_set(self, i) -> dclasses.Equation:
        np.random.seed(i)
        while True:
            try:
                res = self.create_lambda(np.random.randint(2 ** 32 - 1))
                if res == []:
                    continue
                assert type(res) == dclasses.Equation
                return res
            except TimeoutError:
                # signal.alarm(0)
                continue
            except generator.NotCorrectIndependentVariables:
                # signal.alarm(0)
                continue
            except generator.UnknownSymPyOperator:
                # signal.alarm(0)
                continue
            except generator.ValueErrorExpression:
                # signal.alarm(0)
                continue
            except generator.ImAccomulationBounds:
                # signal.alarm(0)
                continue
            except RecursionError:
                # Due to Sympy
                # signal.alarm(0)
                continue
            except KeyError:
                # signal.alarm(0)
                continue
            except TypeError:
                # signal.alarm(0)
                continue
            except Exception as E:
                continue

    def create_lambda(self, i):
        prefix, variables = self.env.generate_equation(np.random)
        prefix = self.env.add_identifier_constants(prefix)
        consts = self.env.return_constants(prefix)
        infix, _ = self.env._prefix_to_infix(prefix, coefficients=self.env.coefficients, variables=self.env.variables)
        consts_elemns = {y: y for x in consts.values() for y in x}
        # constants_expression = infix.format(**consts_elemns)
        w_const, wout_consts = data_utils.sample_symbolic_constants_from_coeff_dict(consts_elemns,
                                                                                    self.cfg.dataset_train.constants)
        eq_string = infix.format(**w_const)

        try:
            eq_infix = str(sympy.sympify(eq_string))
            if 'zoo' in eq_infix or 'nan' in eq_infix or "I" in eq_infix or "E" in eq_infix or "pi" in eq_infix:
                return []
        except:
            return []

        try:
            eq_sympy_infix = constants_to_placeholder(eq_string)
            eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)
        except UnknownSymPyOperator as e:
            print(e)
            return []
        except RecursionError as e:
            print(e)
            return []

        res = dclasses.Equation(expr_infix=infix, coeff_dict=consts_elemns, variables=variables,
                                pre_order_traversal=eq_sympy_prefix)

        # signal.alarm(0)
        return res


@click.command()
@click.option(
    "--number_of_expressions",
    default=200,
    help="Total number of expressions to generate",
)
@click.option(
    "--eq_per_block",
    default=5e4,
    help="Total number of expressions to generate",
)
@click.option("--debug/--no-debug", default=False)
def creator(number_of_expressions, eq_per_block, debug):
    copyreg.pickle(types.CodeType, code_pickler, code_unpickler)  # Needed for serializing code objects
    total_number = number_of_expressions
    cpus_available = multiprocessing.cpu_count()
    eq_per_block = min(total_number // cpus_available, int(eq_per_block))
    print("There are {} expressions per block. The progress bar will have this resolution".format(eq_per_block))
    warnings.filterwarnings("error")
    env, param, config_dict = create_env("./dataset_configuration.json")
    if not debug:
        folder_path = Path(f"../data/raw_datasets/{number_of_expressions}")
    else:
        folder_path = Path(f"../data/raw_datasets/{number_of_expressions}")
    h5_creator = H5FilesCreator(target_path=folder_path)
    env_pip = Pipepile(env,
                       number_of_expressions=number_of_expressions,
                       eq_per_block=eq_per_block,
                       h5_creator=h5_creator,
                       is_timer=not debug)
    starttime = time.time()

    if not debug:
        try:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                max_ = total_number
                with tqdm(total=max_) as pbar:
                    for f in p.imap_unordered(env_pip.create_block, range(0, total_number, eq_per_block)):
                        pbar.update()

        except:
            print(traceback.format_exc())


    else:
        list(map(env_pip.create_block, tqdm(range(0, total_number, eq_per_block))))

    dataset = dclasses.DatasetDetails(
        config=config_dict,
        total_coefficients=env.coefficients,
        total_variables=list(env.variables),
        word2id=env.word2id,
        id2word=env.id2word,
        una_ops=env.una_ops,
        bin_ops=env.una_ops,
        rewrite_functions=env.rewrite_functions,
        total_number_of_eqs=number_of_expressions,
        eqs_per_hdf=eq_per_block,
        generator_details=param)
    print("Expression generation took {} seconds".format(time.time() - starttime))
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    t_hf = h5py.File(os.path.join(folder_path, "metadata.h5"), 'w')
    t_hf.create_dataset("other", data=np.void(pickle.dumps(dataset)))
    t_hf.close()


if __name__ == "__main__":
    creator()
