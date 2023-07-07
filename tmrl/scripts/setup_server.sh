#!/bin/bash

display_width=900
display_height=600
display_number=98
vnc_local_port=42420

while getopts "p:" opt; do
  case $opt in
    p)
      vnc_local_port=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

export DISPLAY=:$display_number

sudo pkill Xvfb
sudo fuser -k $vnc_local_port/tcp
tmux kill-session -t lutris
tmux kill-session -t vnc-server
tmux kill-session -t xvfb-server

echo "CLEANUP SUCCESSFUL; START SETUP"

tmux new-session -d -s xvfb-server "Xvfb :$display_number -ac -screen 0 ${display_width}x${display_height}x24"
tmux new-session -d -s vnc-server "x11vnc -rfbport $vnc_local_port -display :$display_number -localhost"
tmux new-session -d -s lutris 'lutris'
