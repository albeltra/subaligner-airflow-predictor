# Subaligner Ubuntu 20 Docker Image
FROM ubuntu:22.04 

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

RUN ["/bin/bash", "-c", "apt-get -y update &&\
    apt-get -y install ffmpeg &&\
    apt-get -y install espeak libespeak1 libespeak-dev espeak-data &&\
    apt-get -y install libsndfile-dev"]

RUN apt-get install -y wget git

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh &&\
    chmod +x Miniconda3-latest-Linux-x86_64.sh &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN conda install -c conda-forge gxx

RUN wget -O /usr/share/keyrings/gpg-pub-moritzbunkus.gpg https://mkvtoolnix.download/gpg-pub-moritzbunkus.gpg

RUN apt update

RUN apt install -y mkvtoolnix

RUN git clone https://github.com/baxtree/subaligner.git /subaligner

RUN cd /subaligner && python3 -m pip install -e.

COPY ./subaligner-trained/subaligner/models/training/weights/weights.hdf5 /subaligner/subaligner/models/training/weights/

COPY ./subaligner-trained/subaligner/models/training/model/model.hdf5 /subaligner/subaligner/models/training/model/

RUN mkdir -p /airflow/xcom/

COPY ./predict.py /scripts/

RUN mv /subaligner/subaligner/predictor.py /subaligner/subaligner/old_predictor.py

COPY ./predictor.py /subaligner/subaligner/predictor.py

COPY ./network.py /subaligner/subaligner/network.py

RUN mv /subaligner/subaligner/subaligner_1pass/__main__.py /subaligner/subaligner/subaligner_1pass/old__main__.py

COPY ./__main__.py /subaligner/subaligner/subaligner_1pass/__main__.py