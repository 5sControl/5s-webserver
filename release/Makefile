.PHONY: run

run:
	sudo mkdir -p /home/server/reps/images/
	sudo mkdir -p /home/server/reps/videos/
	sudo mkdir -p /home/server/reps/database/
	sudo mkdir -p /home/server/reps/database/pgdata
	sudo docker pull 5scontrol/machine_control_python:1.0.0 && sudo docker pull 5scontrol/operation_control_js:v1.0.0-rc.2 && sudo docker pull 5scontrol/min_max_python:v0.3.12 && sudo docker pull 5scontrol/idle_python:v0.3.11 && sudo docker-compose pull && sudo docker-compose up
