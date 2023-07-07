#!/bin/bash

username="kpius"
servername="ee-tik-vm050.ethz.ch"

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

ssh -L 9999:localhost:5566 "$username@$servername"

