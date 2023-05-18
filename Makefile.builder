VERSION ?= v0.3.6-legendary

build:
	sudo mkdir -p /home/server/reps/images/
	sudo mkdir -p /home/server/reps/videos/
	sudo mkdir -p /home/server/reps/database/
	sudo mkdir -p /home/server/reps/database/pgdata

	download_files:
	curl -LJO https://github.com/5sControl/server-/releases/download/$(VERSION)/docker-compose.yml
	curl -LJO https://github.com/5sControl/server-/releases/download/$(VERSION)/Makefile
	mv docker-compose.yml /home/server/reps/
	mv Makefile /home/server/reps/
