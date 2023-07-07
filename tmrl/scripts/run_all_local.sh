#!/bin/bash

echo "Starting tmrl-server..."
tmux new-session -d -s tmrl-server "python -m tmrl --server"
sleep 2

echo "Starting tmrl-trainer..."
tmux new-session -d -s tmrl-trainer "python -m tmrl --trainer"
sleep 2

echo "Starting tmrl-worker..."
tmux new-session -s tmrl-worker "python -m tmrl --worker"
