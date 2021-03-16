FROM tiangolo/uwsgi-nginx-flask:python3.8 as base

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

FROM base as take_home_project
COPY . /usr/src/predictor
WORKDIR /usr/src/predictor

ENV METAFLOW_USER="Dockerized_user"

ENTRYPOINT [ "python" ]
CMD [ "pipeline.py" , "run"]