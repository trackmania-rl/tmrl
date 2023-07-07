#!/bin/bash

display_width=900
display_height=600
display_number=98
vnc_local_port=5566
start_vnc_server=false

while getopts "p:" opt; do
  case $opt in
    p)
      vnc_local_port=$OPTARG
      start_vnc_server=true
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

echo "CLEANUP SUCCESSFULL; START SETUP"

tmux new-session -d -s xvfb-server "Xvfb :$display_number -ac -screen 0 ${display_width}x${display_height}x24"
if [ "$start_vnc_server" = true ] ; then
  tmux new-session -d -s vnc-server "x11vnc -rfbport $vnc_local_port -display :$display_number -localhost"
fi

tmux new-session -d -s lutris 'lutris'
