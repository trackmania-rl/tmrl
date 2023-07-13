#!/bin/bash

echo "start tmux-session vnc-server for display $DISPLAY on port $PORT"
x11vnc -display $DISPLAY -rfbport $PORT -forever -shared &

