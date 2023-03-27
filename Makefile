.PHONY: django front algorithms pull-all --run

django:
	cd ../5sControll-backend-django/ && git pull && sudo docker build -t artsiom24091/django . && cd ../server-
front:
	cd ../django-front/ && git pull && sudo docker build -t artsiom24091/5scontrol_front . && cd ../server-
algorithms:
	cd ../algorithms/ && git pull && sudo docker build -t artsiom24091/algorithms . && cd ../server-
onvif:
	cd ../onvif/ && git pull && sudo docker build -t artsiom24091/onvif . && cd ../server-
pull-all:
	make django
	make front
	make algorithms
	make onvif
run:
	sudo docker-compose down && sudo docker-compose up
