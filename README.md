<p align="center">
  <img src="https://github.com/5sControl/.github/assets/131950264/d65c5be1-1cca-46a1-b564-ef4364f0dbc6" alt="5controlS" height = "320" />
</p>

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

**[5controlS](https://5controls.com/)** is an open source AI-powered manufacturing operations platform that helps companies improve productivity, eliminate waste, and enhance workplace safety using computer vision (CV) and machine learning (ML). 

The platform generates detailed reports to provide insights on how processes can be optimized.

Plug-in your ML-algorithms or some of the Official 5s algorithms and get a ready-to-use software for manufacturing monitoring.

# **Project repositories**

The connections between the project repositories are illustrated by the following diagram. 

> Please note that to ensure system stability and perfomance you can use one of the Official 5S algorithms instead of Your Algorithm.

<p align="center">
  <img src="https://github.com/5sControl/5s-webserver/assets/131950264/6bfb9a7b-e16f-4600-8656-3f41fc61dba3" alt="5controlS-diagram" />
</p>


**5controlS Platform:**
1. [5s-backend](https://github.com/5sControl/5s-backend)
2. [5s-frontend](https://github.com/5sControl/5s-frontend)
3. [5s-algorithms-controller](https://github.com/5sControl/5s-algorithms-controller)
4. [5s-onvif](https://github.com/5sControl/5s-onvif)
5. [5s-onvif-finder](https://github.com/5sControl/5s-onvif-finder)

**Official Algorithms:**
1. [min-max](https://github.com/5sControl/min-max)
2. [idle-control](https://github.com/5sControl/idle-control)
3. [operation-control-js](https://github.com/5sControl/operation-control-js)
4. [machine-control](https://github.com/5sControl/machine-control)
5. [machine-control-js](https://github.com/5sControl/machine-control-js)

**Algorithms Servers:**
1. [inference-server-js]()

# **Documentation**

[User Documentation](https://github.com/5sControl/Manufacturing-Automatization-Enterprise/wiki)

[Installation Guide Pt.1](https://5controls.com/blog/installing-5s-control-software)

[Installation Guide Pt.2](https://5controls.com/blog/installing-5s-control-software-part-2-docker)

# **Installation**
To run the application, you need to install Docker and Docker-compose. You can do this by following the instructions specific to your system available at [Docker Installation Guide](https://docs.docker.com/engine/install/) [Docker-compose Installation Guide](https://docs.docker.com/compose/install/).

Please follow the link and select the instructions that match your operating system to install Docker.

To clone the project, use the following Git command:

```shell
git@github.com:5sControl/5s-webserver.git
```

## **Docker-compose**

### **Django:**

#### **Environment variables**

You can view all environment variables in the service [repository](https://github.com/5sControl/5s-backend).

#### **Volumes**

 - `/home/server/reps/images:/usr/src/app/images` - to access platform images
 - `/home/server/reps/videos:/usr/src/app/videos` to access platform videos
 - `/home/server/reps/log:/usr/src/app/log` - to access django logs

**DB:**

#### **Environment variables**

 - `POSTGRES_USER: user` - Database user
 - `POSTGRES_PASSWORD: pass` - Database password
 - `POSTGRES_DB: db` - Database name

#### **Volumes**

 - `/home/server/reps/database/pgdata:/var/lib/postgresql/data/` - to save the database on device

### **Onvif:**

#### **Environment variables**

You can view all environment variables in the service [repository](https://github.com/5sControl/5s-onvif).

#### **Volumes**

 - `/home/server/reps/images:/usr/src/app/images` - to save pictures on the device
 - `/home/server/reps/videos:/usr/src/app/videos` to save videos on the device
 - `/home/server/reps/database:/var/www/5scontrol/database` - to save Sqlite Database on device

### **Algorithms controller:**

#### **Environment variables**

You can view all environment variables in the service [repository](https://github.com/5sControl/5s-algorithms-controller).

#### **Volumes**
 - `/home/server/reps/images:/var/www/5scontrol/images` - to save report pictures on the device
 - `/home/server/reps/database/dataset:/var/www/5scontrol/dataset` - to save dataset on the device
 - `/home/server/reps/images/debug:/var/www/5scontrol/debug` - to save debug pictures on the device
 - `/var/run/docker.sock:/var/run/docker.sock` - to access your docker driver

### **Onvif finder**

#### **Environment variables**

 - `IP: 192.168.1.132` - network ip

### **Tests**

#### **Environment variables**

 - `IP: 192.168.1.132` - network ip
 - `BACKEND_PORT:8000` - backend port

### **Frontend**

#### **Environment variables**

 - `REACT_APP_NGROK: https://example.com` - domain link

### **NGINX**

#### **Volumes**

 - `/home/server/reps/images:/usr/src/app/images` - to access pictures
 - `/home/server/reps/videos:/usr/src/app/videos` to access videos

### **Algorithm Models (Min_Max, Idle, Machine_control)**

#### **Environment variables**

 - `server_url: 192.168.1.142` - network ip

##### **Machine_control_JS**

- `link_reports: http://${network ip}:8000/api/reports/report-with-photos/` - link to send reports to the backend

#### **Resources**

To change the maximum model memory you need to change the resource parameter

![image](https://github.com/5sControl/5s-webserver/assets/57481641/20bb6a92-b644-4ae1-9b70-7e4fd2b6e813)

**Running the Application:**

For start, use the following command: 
 ```shell
 cd release
 docker-compose up
```

# **Contributing**
Thank you for considering contributing to 5controlS. We truly believe that we can build an outstanding product together!

We welcome a variety of ways to contribute. Read below to learn how you can take part in improving 5controlS.

## **Code of conduct**

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## Code contributing

If you want to contribute, read  our [contributing guide](CONTRIBUTING.md) to learn about our development process and pull requests workflow.

We also have a list of [good first issues](https://github.com/5sControl/5s-webserver/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) that will help you make your first step to beсoming a 5S contributor.

# **License**
[GNU GPL-3.0](LICENSE.md)

> 5controlS uses third party libraries that are distributed under their own terms (see [LICENSE-3RD-PARTY.md](https://github.com/5sControl/5s-webserver/blob/main/LICENSE-3RD-PARTY.md)).<br>

<br>
<div align="center">
  <a href="https://5controls.com/" style="text-decoration:none;">
    <img src="https://github.com/5sControl/Manufacturing-Automatization-Enterprise/blob/3bafa5805821a34e8b825df7cc78e00543fd7a58/assets/Property%201%3DVariant4.png" width="10%" alt="" /></a> 
  <img src="https://github.com/5sControl/5s-backend/assets/131950264/d48bcf5c-8aa6-42c4-a47d-5548ae23940d" width="3%" alt="" />
  <a href="https://github.com/5sControl" style="text-decoration:none;">
    <img src="https://github.com/5sControl/Manufacturing-Automatization-Enterprise/blob/3bafa5805821a34e8b825df7cc78e00543fd7a58/assets/github.png" width="4%" alt="" /></a>
  <img src="https://github.com/5sControl/5s-backend/assets/131950264/d48bcf5c-8aa6-42c4-a47d-5548ae23940d" width="3%" alt="" />
  <a href="https://www.youtube.com/@5scontrol" style="text-decoration:none;">
    <img src="https://github.com/5sControl/Manufacturing-Automatization-Enterprise/blob/ebf176c81fdb62d81b2555cb6228adc074f60be0/assets/youtube%20(1).png" width="5%" alt="" /></a>
</div>
