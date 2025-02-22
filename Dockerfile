FROM python:3.9-alpine

WORKDIR /app
VOLUME /data
RUN pip3 install deepanything --upgrade

CMD ["python3", "-m","deepanything","--config","/data/config.json"]