.PHONY: run

run:
	sudo mkdir -p /home/server/reps/images/
	sudo mkdir -p /home/server/reps/videos/
	sudo mkdir -p /home/server/reps/database/
	sudo mkdir -p /home/server/reps/database/pgdata
	sudo mkdir -p /home/server/reps/debug
	sudo docker pull 5scontrol/machine_control_python:v1.2.5 && sudo docker pull 5scontrol/operation_control_js:v1.0.0-rc.6 && sudo docker pull 5scontrol/min_max_python:v0.4.3 && sudo docker pull 5scontrol/idle_python:v0.4.3 && sudo docker-compose pull && sudo docker-compose up
