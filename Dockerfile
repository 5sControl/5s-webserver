FROM python:3.9
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 cmake -y
COPY requirements_headless.txt .
COPY requirements_gpu.txt .
RUN pip install -r requirements_headless.txt
RUN pip install -r requirements_gpu.txt
WORKDIR /var/www/5scontrol
COPY . .
# tell the port number the container should expose
EXPOSE 3010

RUN mkdir -p /usr/src/app

ADD entrypoint.sh /usr/src/app/

RUN ["chmod", "+x", "/usr/src/app/entrypoint.sh"]

# run the command
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
