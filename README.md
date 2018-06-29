# **Distributed Deep Reinforcement Learning** with
# **pytorch** & **tensorboard**
*******

* Sample on-line plotting while training a Distributed DQN agent on Pong:
    * blue: `num_actors=2, nstep=1`
    * orange: `num_actors=8, nstep=1`
    * grey: `num_actors=8, nstep=5`
![dqn_pong](/assets/dqn_pong.png)
*******

## What is included?
This repo currently contains the following agents:

- Distributed DQN [[1]](https://openreview.net/forum?id=H1Dy---0Z)
- Distributed DDPG [[2]](https://openreview.net/forum?id=SyZipzbCb)
*******

## Code structure:
NOTE: we follow the same code structure as [pytorch-rl](https://github.com/jingweiz/pytorch-rl)& [pytorch-dnc](https://github.com/jingweiz/pytorch-dnc).
* ```./utils/factory.py```
> We suggest the users refer to ```./utils/factory.py```,
 where we list all the integrated ```Env```, ```Model```,
 ```Memory```, ```Agent``` into ```Dict```'s.
 All of those four core classes are implemented in ```./core/```.
 The factory pattern in ```./utils/factory.py``` makes the code super clean,
 as no matter what type of ```Agent``` you want to train,
 or which type of ```Env``` you want to train on,
 all you need to do is to simply modify some parameters in ```./utils/options.py```,
 then the ```./main.py``` will do it all (NOTE: this ```./main.py``` file never needs to be modified).
* ```./core/single_processes/.```
> Each agent contains ```4``` types of ```single_process```'s:
> * `Logger`: plot `Global/Actor/Learner/EvaluatorLogs` onto `tensorboard`
> * `Actor`: collect experiences from `Env` and push to a global shared `Memory`
> * `Learner`: samples from the global shared `Memory` and do DRL updates on the `Model`
> * `Evaluator`: evaluate the `Model` during training
*******

## How to run:
You only need to modify some parameters in ```./utils/options.py``` to train a new configuration.

* Configure your training in ```./utils/options.py```:
> * ```line 10```: add an entry into ```CONFIGS``` to define your training (```agent_type```, ```env_type```, ```game```, ```memory_type```, ```model_type```)
> * ```line 23```: choose the entry you just added
> * ```line 19-20```: fill in your machine/cluster ID (```MACHINE```) and timestamp (```TIMESTAMP```) to define your training signature (```MACHINE_TIMESTAMP```),
 the corresponding model file of this training will be saved under this signature (```./models/MACHINE_TIMESTAMP.pth``` ).
 Also the tensorboard visualization will be displayed under this signature (first activate the tensorboard server by type in bash: ```tensorboard --logdir logs/```, then open this address in your browser: ```http://localhost:6006/```)
> * ```line 22```: to train a model, set ```mode=1``` (training visualization will be under ```http://localhost:6006/```); to test the model of this current training, all you need to do is to set ```mode=2``` .

* Run:
> ```python main.py```
*******


## Dependencies:
- Python 3
- [PyTorch >=v0.4.0](http://pytorch.org/)
- [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch)
- [OpenAI Gym](https://github.com/openai/gym)
*******


## Repos we referred to during the development of this repo:
* [Kaixhin/Rainbow](https://github.com/Kaixhin/Rainbow)
* [dgriff777/rl_a3c_pytorch](https://github.com/dgriff777/rl_a3c_pytorch)
* [ShangtongZhang/DeepRL]( https://github.com/ShangtongZhang/DeepRL)

This repo is developed together w/ [@onlytailei](https://github.com/onlytailei).
*******
