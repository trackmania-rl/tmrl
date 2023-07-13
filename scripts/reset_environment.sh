#!/bin/bash

echo "kill old processes ..."
sudo pkill Xvfb
sudo fuser -k $PORT/tcp
tmux kill-session -t lutris
tmux kill-session -t xvfb-server
