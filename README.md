# Transformer-based model for symbolic regression via joint supervised learning

<p align="center">
<img src="overview.png" width=750/>
</p>

This repository contains the official Pytorch implementation for the paper "Transformer-based model for symbolic regression via joint supervised learning" [ICLR 2023]. 

[Paper](https://openreview.net/forum?id=ULzyv9M1j5)         [ICLR 2023](https://iclr.cc/virtual/2023/poster/10690)

# Getting started

- Install Anaconda and create a new environment

- Install the following packages in turn:

  ```python
  pip install numpy
  pip install sympy
  pip install torch
  pip install multiprocessing
  pip install pickle
  pip install h5py
  pip install pathlib
  pip install omegaconf
  pip install tqdm
  pip install gplearn
  pip install glob
  pip install json
  pip install functools
  pip install hydra
  pip install transformers
  ```



# Getting datasets

First, if you want to change the defaults value, configure the `dataset_configuration.json` file:

```
{
    "max_len": 20,
    "operators": "add:10,mul:10,sub:5,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,log:4,exp:4,sin:4,cos:4,tan:4",
    "max_ops": 5,
    "rewrite_functions": "",
    "variables": ["x_1", "x_2"],
    "eos_index": 1,
    "pad_index": 0
}
```

## Generate expressions skeletons base

- You can run this script to generate expressions skeletons:

  ```
  python data_creation/dataset_creation.py --number_of_expressions NumberOfExpressions --no-debug #Replace NumberOfExpressions with the number of expressions you want to generate
  ```

## Generate datasets with different numbers of expressions with different constants

### Generate training set

- Based on the first step, you can run `add_points_to_json.py` and set `number_per_equation` to the number of expressions with different constants. The dataset will be saved as a JSON file. Each sample in the file contains the data points, the skeleton, the first-order traversal list, and the expression itself.

### Generate validation set

- You can randomly select 1000 expression skeletons from the skeleton base, then assign different constants through running `add_points_to_json.py` and save it as a JSON file.

### Generate SSDNC benchmark

- Run following script to generate SSDNC benchmark:

  ```
  python gen_SSDNC_benchmark.py
  ```

  

#  DDP Training

You can configure the `config.yaml` as you choose. If you only have singe GPU, you will need to comment out the DDP code in `train_pytorch.py`.

You can run the following script to train the model:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_pytorch.py
```



# References

- https://github.com/mojivalipour/symbolicgpt
- https://github.com/brendenpetersen/deep-symbolic-optimization
- https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales
- https://github.com/trevorstephens/gplearn
- https://github.com/ma-xu/pointMLP-pytorch



# Citing this work

If you found our work useful and used code, please use the following citation:

```
@inproceedings{litransformer,
  title={Transformer-based model for symbolic regression via joint supervised learning},
  author={Li, Wenqiang and Li, Weijun and Sun, Linjun and Wu, Min and Yu, Lina and Liu, Jingyi and Li, Yanjie and Tian, Songsong},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```



