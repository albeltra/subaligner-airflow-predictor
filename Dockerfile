# Subaligner Ubuntu 20 Docker Image
FROM ubuntu:22.04 

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

RUN ["/bin/bash", "-c", "apt-get -y update &&\
    apt-get -y install ffmpeg &&\
    apt-get -y install espeak libespeak1 libespeak-dev espeak-data &&\
    apt-get -y install libsndfile-dev"]

RUN apt-get install -y wget git

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh &&\
    chmod +x Miniconda3-latest-Linux-x86_64.sh &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN conda install -c conda-forge gxx

RUN wget -O /usr/share/keyrings/gpg-pub-moritzbunkus.gpg https://mkvtoolnix.download/gpg-pub-moritzbunkus.gpg

RUN apt update

RUN apt install -y mkvtoolnix

COPY ./subaligner-trained/ /subaligner/

#RUN apt -y install libblas3 liblapack3 liblapack-dev libblas-dev libatlas-base-dev gfortran

#RUN conda install numpy scipy h5py

RUN cd /subaligner && python3 -m pip install -e.

RUN python3 -m pip install rq==1.12.0 pycountry

RUN mkdir -p /airflow/xcom/

COPY ./predict.py /scripts/

RUN mv /subaligner/subaligner/predictor.py /subaligner/subaligner/old_predictor.py

COPY ./predictor.py /subaligner/subaligner/predictor.py

RUN mv /subaligner/subaligner/subaligner_1pass/__main__.py /subaligner/subaligner/subaligner_1pass/old__main__.py

COPY ./__main__.py /subaligner/subaligner/subaligner_1pass/__main__.py
