# Goal Representations for Instruction Following (GRIF)

This is the code repository for the paper *Goal Representations for Instruction Following: A Semi-Supervised Language Interface to Control* [[arXiv](https://arxiv.org/abs/2307.00117), [website](https://rail-berkeley.github.io/grif/)]. 

Based on [dibyaghosh/jaxrl_minimal](https://github.com/dibyaghosh/jaxrl_minimal).


## Environment
For GPU:
```
conda env create -f environment_cuda.yml
```

For TPU:
```
conda env create -f environment_tpu.yml
```

See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax. 

## Running

To train GRIF, run
```
bash experiments/scripts/launch_bridge.sh GRIF
```
