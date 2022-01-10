# Tutorial: TMRL as a Python library

In the previous sections, we have seen how to use `tmrl` as a standalone program for simple tasks:
you could just modify the `config.json` file according to your needs and launch the server, trainer and rollour workers with simple commands such as `python -m tmrl --worker`.

However, as soon as you will need to do more advanced things (e.g., using robots, other video games, other training algorithms, etc...), you will want to get your hands dirty with some python coding.
This is when you need to start using `tmrl` as a python library.

## Constants
The constants defined in the `config.json` file are accessible via the [config_constants](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/config/config_constants.py) module.
This module can be imported in your script as follows:
```python
import tmrl.config.config_constants as cfg
```
You can then use the constants from this module in your script, e.g.:

```python
print(f"Run name: {cfg.RUN_NAME}")
```


## Trainer
Training in `tmrl` is done within a [TrainingOffline](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/training_offline.py) object, while network communications are handled by a [TrainerInterface](https://github.com/trackmania-rl/tmrl/blob/58f66a42ea0e1478641336fa1eb076635ff77a31/tmrl/networking.py#L389).

These two objects are used 

## Rollout worker(s)


## Server