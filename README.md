
# Medical Overseer Control (MOC) New ML API

This is the new ML Serving api that's replacing the old one. It improves the model accuracy by reworking and retraining the old model. We're using GCP's compute engine  this time instead of app engine because of its flexibility and lack of time for implementing a correct docker solution


## Deployment
1. make an ubuntu-based compute engine instance

2. create an ingress firewall rule in the gcp console to allow port 8080

3. SSH into the instance

4. install git, unzip, and pip using this command
```bash
$ sudo apt-get install git unzip python3-pip libgl1 -y
```
5. clone the tensorflow repo using
``` bash
$ git clone https://github.com/tensorflow/models.git
```

6. enter the models folder
```bash
$ cd ~/models/
```
7. make a directory inside the models folder called protocs with
```bash
$ mkdir protoc
$ cd protoc
```

8. download protocs and unzip
```bash
$ wget https://github.com/protocolbuffers/protobuf/releases/download/v23.3/protoc-23.3-linux-x86_64.zip
$ unzip protoc-23.3-linux-x86_64.zip
```
9. go into the models/research folders and compile the tensorflow object detection api using protoc
```bash
$ cd ~/models/research
$ ~/models/protoc/bin/protoc object_detection/protos/*.proto --python_out=.
```
10. install tensorflow object detection api dependencies
```bash
$ cp object_detection/packages/tf2/setup.py .
$ python3 -m pip install --use-feature=2020-resolver .
```

11. clone our api and model
```bash
$ git clone
```
12. copy to base models/research folder
```bash
$ cp ~/models/research/moc_newmlserving_api/*  ~/models/research
```
13. install dependencies
```bash
$ pip install -r requirements.txt
```
14. start the server
```bash
$ gunicorn --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind :8080 app1:app
```
