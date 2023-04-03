.PHONY: django front algorithms onvif pull-all run push clear-images

django:
	cd ../5sControll-backend-django/ && git pull && sudo docker build -t artsiom24091/django${version} . && cd ../server-
front:
	cd ../django-front/ && git pull && sudo docker build -t artsiom24091/5scontrol_front${version} . && cd ../server-
algorithms:
	cd ../algorithms/ && git pull && sudo docker build -t artsiom24091/algorithms${version} . && cd ../server-
onvif:
	cd ../onvif/ && git pull && sudo docker build -t artsiom24091/onvif${version} . && cd ../server-
pull-all:
	make django
	make front
	make algorithms
	make onvif
run:
	sudo docker-compose down && sudo docker-compose up
push:
	sudo docker push artsiom24091/django${version} && sudo docker push artsiom24091/onvif${version} && sudo docker push artsiom24091/algorithms${version} && sudo docker push artsiom24091/5scontrol_front${version}
clear-images:
	docker rmi $(docker images -f "dangling=true" -q)
