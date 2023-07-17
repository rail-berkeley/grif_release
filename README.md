# Goal Representations for Instruction Following (GRIF)

This is the code repository for the paper *Goal Representations for Instruction Following: A Semi-Supervised Language Interface to Control* [[arXiv](https://arxiv.org/abs/2307.00117), [website](https://rail-berkeley.github.io/grif/)]. 

Based on [dibyaghosh/jaxrl_minimal](https://github.com/dibyaghosh/jaxrl_minimal).


## Environment
```
conda create -n jaxrl python=3.10
conda activate jaxrl
pip install -e . 
pip install -r requirements.txt
```
For GPU:
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU
```
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax. 
