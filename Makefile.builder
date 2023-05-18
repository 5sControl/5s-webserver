VERSION ?= v0.3.6-legendary

mkdir:
	sudo mkdir -p /home/server/reps/images/
	sudo mkdir -p /home/server/reps/videos/
	sudo mkdir -p /home/server/reps/database/
	sudo mkdir -p /home/server/reps/database/pgdata

download_files:
	curl -LJO https://github.com/5sControl/server-/releases/download/$(VERSION)/docker-compose.yml
	curl -LJO https://github.com/5sControl/server-/releases/download/$(VERSION)/Makefile
	mv docker-compose.yml /home/server/reps/
	mv Makefile /home/server/reps/

edit_docker_compose:
	IP := $(shell hostname -I | grep -Eo '192\.[0-9]+\.[0-9]+\.[0-9]+')
	sed -i 's/SERVER_URL:.*/SERVER_URL: "http:\/\/$(IP)"/' /home/server/reps/docker-compose.yml
	sed -i 's/IP:.*/IP: "$(IP)"/' /home/server/reps/docker-compose.yml

run_server:
	cd /home/server/reps/
	make -f Makefile run

build:
	make -f Makefile.builder mkdir
	make -f Makefile.builder download_files
	make -f Makefile.builder edit_docker_compose
	make -f Makefile.builder run_server
