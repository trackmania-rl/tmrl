# Getting started with TMRL

This document provides a brief intro of the usage of builtin command-line tools in TMRL, how to infer the self driving car trackmania map and how to train a car on whatever map you have choosen.
Then We will how you how to use the API in other games.


# Inference Demo with existing models

By inference, we mean using trained models to drive the car, the each models are trainned for 10 hours on differents tracks.
In TMRL the weights of the model are alawys saved at 'data/weights'

TMRL provides 2 self driving car models, the car will either use a LIDAR or raw images to perceive its envoronement.

## set the game
first, you will need to set the game, to do that you have to put the windows game at the top left on your screenm you can bang the window to be sure and then resize the window at 958/488 for lidar and 256/128 for the camera
If you want to use the lidar you will use `camera game 3` and `player visibility for cockpit view = old`
Butm if you prefer to use the camera you cam use `camera game 1`  
In both case you should remove the interface (time, ...)

To run the Lidar, go to tmrl, custom, config and set `PRAGMA_LIDAR = True`
for the camera you can set to `False`

`PRAGMA_CUDA_INFERENCE = False` if you want to use the gpu
then you can run the following command:
```shell
Python tmrl/run.py --test
```
then you can run this command and click directly on the game windows with your mouse

# Train a new car
 Now we will see how to train our own car with the track you want

if you want to train the car with the lidar you will need to train it on road only, the lidar doesn't deal with other surfaces.

You can train the car on whatever surfaces if it is trained with camera but you computer may need a good GPU to run both CNN and trackmania.

# Use TMRL APIs in your favorite game

you can simply run the code on tmnf by changing this option in config.py: `PRAGMA_TM2020_TMNF = True`


