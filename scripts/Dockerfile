FROM nvidia/cuda:12.2.0-base-ubuntu22.04

MAINTAINER Pius 

RUN apt-get update

RUN apt-get install -y sudo wget curl vim
RUN apt-get install -y xvfb
RUN apt-get install -y x11vnc
RUN apt-get install -y tmux
RUN apt-get install -y lutris

RUN dpkg --add-architecture i386 
RUN apt-get update
RUN apt-get -y install wine32

RUN apt-get -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y install xdotool
RUN apt-get -y install python3-pip
RUN apt-get -y install libx11-dev
RUN pip install numpy
RUN pip install fastgrab
RUN pip install opencv-python

ARG PARAM_UID=8500
ARG PARAM_GID=8500

RUN groupadd -g "${PARAM_GID}" "vncuser" || true
RUN adduser --disabled-password --uid "${PARAM_UID}" --gid "${PARAM_GID}" --gecos "vncuser" "vncuser"
RUN groupadd -r supersudo
RUN echo "%supersudo ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/supersudo
RUN usermod -a -G supersudo "vncuser"

ADD start_server.sh /home/vncuser/start_server.sh
ADD setup_environment.sh /home/vncuser/setup_environment.sh
ADD reset_environment.sh /home/vncuser/reset_environment.sh
ADD install_trackmania.py /home/vncuser/install_trackmania.py
RUN chmod +x home/vncuser/start_server.sh
RUN chmod +x home/vncuser/setup_environment.sh
RUN chmod +x home/vncuser/reset_environment.sh
RUN chmod +x home/vncuser/install_trackmania.py

USER vncuser
ENV HOME /home/vncuser
ENV USER vncuser
WORKDIR /home/vncuser

ARG port=5900
ENV PORT=$port
EXPOSE $port

CMD "/bin/bash"
