.PHONY: run

run:
	sudo mkdir -p /home/server/reps/images/
	sudo mkdir -p /home/server/reps/videos/
	sudo mkdir -p /home/server/reps/database/
	sudo mkdir -p /home/server/reps/database/pgdata
	sudo docker pull 5scontrol/min_max_python:v0.3.10 && sudo docker pull 5scontrol/idle_python:v0.3.9 && sudo docker-compose pull && sudo docker-compose up
