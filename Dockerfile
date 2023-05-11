# Subaligner Ubuntu 20 Docker Image
FROM ubuntu:22.04 

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

RUN ["/bin/bash", "-c", "apt-get -y update &&\
    apt-get -y install ffmpeg &&\
    apt-get -y install espeak libespeak1 libespeak-dev espeak-data &&\
    apt-get -y install libsndfile-dev"]

RUN apt-get install -y wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh &&\
    chmod +x Miniconda3-latest-Linux-x86_64.sh &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN conda install -c conda-forge gxx

RUN wget -O /usr/share/keyrings/gpg-pub-moritzbunkus.gpg 
https://mkvtoolnix.download/gpg-pub-moritzbunkus.gpg

RUN apt update

RUN apt install -y mkvtoolnix

COPY ./subaligner-trained/ /subaligner

RUN cd /subaligner && python3 -m pip install -e.

RUN python3 -m pip install rq==1.12.0 pycountry

RUN mkdir -p /airflow/xcom/
