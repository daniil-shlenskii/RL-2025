# Reinforcement Learning course projects


## \#1: Q-Learning

Description of the MDP and the Agent is in the [slides/1\_tabular_slides.pdf](slides/1_qlearning_slides.pdf).

Code of the MDP is in the [src/envs/custom_envs/trex\_1d\_env.py](src/envs/custom_envs/trex_1d_env.py).


Code of the Agent is in the [src/agents/tabular/qlearning.py](src/agents/tabular/qlearning.py).

Train and evaluation is run with the following code:

```shell
python -m run.1_qlearning
```

## \#2: REINFORCE

Description of the MDP and the Agent is in the [slides/2\_reinforce_slides.pdf](slides/2_reinforce_slides.pdf).

Code of the MDP is in the [src/envs/custom_envs/trex\_jump\_env.py](src/envs/custom_envs/trex_jump_env.py).


Code of the Agent is in the [src/agents/general/reinforce.py](src/agents/general/reinforce.py).

Train and evaluation is run with the following code:

```shell
python -m run.2_reinforce
```