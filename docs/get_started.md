# Getting started with TMRL

This document provides a brief intro of the usage of builtin command-line tools in TMRL, how to infer the self driving car trackmania map and how to train a car on whatever map you have choosen.
Then We will how you how to use the API in other games.


# Inference Demo with existing models

By inference, we mean using trained models to drive the car, the each models are trainned for 10 hours on differents tracks.
In TMRL the weights of the model are alawys saved at 'data/weights'

TMRL provides 2 self driving car models, the car will either use a LIDAR or raw images to perceive its envoronement.

you can

then you can run the following command:

```shell
Python tmrl/run.py --test
```

# Train a new car
# Use TMRL APIs in your favorite game
