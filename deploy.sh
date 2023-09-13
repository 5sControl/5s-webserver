#!/bin/bash

DJANGO_URL="git@github.com:5sControl/5sControll-backend-django.git"
DJANGO_IMAGE_NAME="artsiom24091/django"

FRONT_URL="git@github.com:5sControl/django-front.git"
FRONT_IMAGE_NAME="artsiom24091/5scontrol_front"

ALGORITHMS_URL="git@github.com:5sControl/algorithms.git"
ALGORITHMS_IMAGE_NAME="artsiom24091/algorithms"

ONVIF_URL="git@github.com:5sControl/onvif.git"
ONVIF_IMAGE_NAME="artsiom24091/onvif"

while true; do
    is_changes=0
    
    #  check django
    cd ../5sControll-backend-django/
    git fetch
    if ! git diff --quiet HEAD origin/main; then
        git pull
        sudo docker build -t $DJANGO_IMAGE_NAME .
        cd ../server-/
        sudo docker-compose down && sudo docker-compose up --detach
        is_changes=1
    fi
    
    #  check front
    cd ../django-front/
    git fetch
    if ! git diff --quiet HEAD origin/development; then
        git pull
        sudo docker build -t $FRONT_IMAGE_NAME .
        cd ../server-/
        sudo docker-compose down && sudo docker-compose up --detach
        is_changes=1
    fi
  
    #  check algorithms
    cd ../algorithms/
    git fetch
    if ! git diff --quiet HEAD origin/main; then
        git pull
        sudo docker build -t $ALGORITHMS_IMAGE_NAME .
        cd ../server-/
        is_changes=1
    fi
   
    # onvif
    cd ../onvif/
    git fetch
    if ! git diff --quiet HEAD origin/main; then
        git pull
        sudo docker build -t $ONVIF_IMAGE_NAME .
        cd ../server-/
        is_changes=1
    fi

    if [ $is_changes -eq 0 ]; then
        echo "No changes"
    else
	echo "Update"
	sudo docker-compose down && sudo docker-compose up --detach
    fi
    
    sleep 10
done

