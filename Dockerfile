FROM tensorflow/tensorflow:2.5.0-gpu

ENV TZ='Asia/Seoul'
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y python3-tk

COPY requirements.txt .
RUN pip3 install -r requirements.txt
