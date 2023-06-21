.PHONY: django front algorithms-controller onvif run pull-all push clear-images

django:
	cd ../5sControll-backend-django/ && git checkout development && git reset --hard origin/development && git pull && sudo docker build -t 5scontrol/django${version} . && cd ../server-
front:
	cd ../django-front/ && git checkout development && git reset --hard origin/development && git pull && sudo docker build -t 5scontrol/5scontrol_front${version} . && cd ../server-
django-build:
	cd ../5sControll-backend-django/ && git checkout main && git reset --hard origin/main && git pull && sudo docker build -t 5scontrol/django${version} . && cd ../server-
front-build:
	cd ../django-front/ && git checkout main && git reset --hard origin/main && git pull && sudo docker build -t 5scontrol/5scontrol_front${version} . && cd ../server-
algorithms-controller:
	cd ../algorithms-controller/ && git checkout main && git reset --hard origin/main && git pull && sudo docker build -t 5scontrol/algorithms-controller${version} . && cd ../server-
onvif:
	cd ../onvif/ && git checkout main && git reset --hard origin/main && git pull && sudo docker build -t 5scontrol/onvif${version} . && cd ../server-
run:
	sudo docker-compose down && sudo docker-compose up --timeout 600
pull-all:
	make django
	make front
	make onvif
	make algorithms-controller
pull-all-build:
	make django-build
	make front-build
	make onvif
	make algorithms-controller
push:
	sudo docker push 5scontrol/django${version} && sudo docker push 5scontrol/onvif${version} && sudo docker push 5scontrol/algorithms-controller${version} && sudo docker push 5scontrol/5scontrol_front${version}
clear-images:
	docker rmi $(docker images -f "dangling=true" -q)
build:
	sudo mkdir -p /home/server/reps/images/
	sudo mkdir -p /home/server/reps/videos/
	sudo mkdir -p /home/server/reps/database/
	sudo mkdir -p /home/server/reps/database/pgdata
