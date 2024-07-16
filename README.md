## Rethinking Decision Transformer via Hierarchical Reinforcement Learning

This is the original implementation for paper



[Rethinking Decision Transformer via Hierarchical Reinforcement Learning](https://openreview.net/forum?id=WsM4TVsZpJ)





#### 1. Create and activate conda environment

```
conda env create -f env.yaml
```

#### 2. Additional installations

① Install the dependencies related to the mujoco-py environment. For more details, see https://github.com/openai/mujoco-py#install-mujoco

② Install D4RL. For more details, see https://github.com/Farama-Foundation/D4RL

#### 3. Running

```
$ cd algorithms
# For V-ADT, first train IQL's Q/V, then train low-level DT
$ python iql.py
$ python V-ADT.py
# For G-ADT, first train HIQL's Q/V/goal, then train low-level DT
$ python hiql.py
$ python G-ADT.py
```

You can also modify the `bash` files to start the training.  For the hyperparameters setting, please refer to the `config` folder and the appendix in the paper.

Please feel free to contact me if you have questions.