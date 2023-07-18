#!/bin/bash

# Default values
PORT=5900
DISPLAY=:98
DISPLAY_WIDTH=900
DISPLAY_HEIGHT=600

# Parse command-line options
while getopts "p:d:w:h:" opt; do
    case $opt in
        p)
            PORT=$OPTARG
            ;;
        d)
            DISPLAY=$OPTARG
            ;;
        w)
            DISPLAY_WIDTH=$OPTARG
            ;;
        h)
            DISPLAY_HEIGHT=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            exit 1
            ;;
    esac
done

export PORT=$PORT
export DISPLAY=$DISPLAY
export DISPLAY_WIDTH=$DISPLAY_WIDTH
export DISPLAY_HEIGHT=$DISPLAY_HEIGHT

# Rest of your script...

