FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt install nano
RUN apt-get install -y python3 
RUN apt-get install -y python3-pip 

RUN pip3 install  numpy
RUN pip3 install  'scikit-learn==1.0.2'
RUN pip3 install  mlxtend
RUN pip3 install  xgboost

ENV PYTHONPATH="/tsdf-ear"

WORKDIR /tsdf-ear

COPY . .
