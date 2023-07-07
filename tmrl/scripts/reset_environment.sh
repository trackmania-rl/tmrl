#!/bin/bash

export DISPLAY=:$display_number

sudo pkill Xvfb
sudo fuser -k $vnc_local_port/tcp
tmux kill-session -t lutris
tmux kill-session -t vnc-server
tmux kill-session -t xvfb-server
tmux kill-session -t tmrl-server
tmux kill-session -t tmrl-trainer
tmux kill-session -t tmrl-worker