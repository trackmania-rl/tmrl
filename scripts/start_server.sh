#!/bin/bash

echo "setup server, env: DISPLAY=$DISPLAY ($DISPLAY_WIDTH x $DISPLAY_HEIGHT), PORT=$PORT ..."
sleep 1
Xvfb $DISPLAY -ac -screen 0 ${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}x24 &
x11vnc -display $DISPLAY -rfbport $PORT -forever -shared &

echo "started vnc-server for display $DISPLAY on port $PORT"
echo "start xvfb-server on display $DISPLAY (${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}x24)"
echo "start tmux-session lutris"
/usr/games/lutris &
