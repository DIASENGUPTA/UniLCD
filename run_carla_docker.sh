#!/bin/bash
NAME=unilcd_carla_server
GPUID=0
CARLA_SCRIPT=./CarlaUE4.sh
DOCKER_NAME=carlasim/carla:0.9.13
CARLA_PORT=2000

if [ "$1" == "start" ]
then
    docker run --rm -d --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$GPUID --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw --name $NAME $DOCKER_NAME /bin/bash $CARLA_SCRIPT -world-port=$CARLA_PORT -RenderOffScreen -quality-level=Epic -nosound
elif [ "$1" == "stop" ]
then
    docker stop $NAME
fi