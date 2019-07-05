# This is a sample Dockerfile you can modify to deploy your own app based on face_recognition

FROM python:3.7-slim-stretch

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS


# The rest of this file just runs an example script.

# If you wanted to use this Dockerfile to run your own app instead, maybe you would do this:
MAINTAINER Practice School 1
ENV PYTHONUNBUFFERED 1
WORKDIR /
COPY . /
RUN pip3 install -r requirements.txt

# RUN adduser -D user
# USER user
RUN python3 manage.py runserver

# COPY . /root/face_recognition
# RUN cd /root/face_recognition && \
#     pip3 install -r requirements.txt && \
#     python3 setup.py install

# CMD cd /root/face_recognition/examples && \
# python3 recognize_faces_in_pictures.py
