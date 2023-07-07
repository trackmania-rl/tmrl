#!/bin/bash

username="kpius"
servername="ee-tik-vm050.ethz.ch"
vnc_local_port=9999
vnc_server_port=42420

while getopts ":u:s:" opt; do
  case $opt in
    u)
      username=$OPTARG
      ;;
    s)
      servername=$OPTARG
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

shift $((OPTIND - 1))

ssh -L "$vnc_local_port":localhost:"$vnc_server_port" "$username@$servername"

