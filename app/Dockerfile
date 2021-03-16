FROM tiangolo/uwsgi-nginx-flask:python3.8 as base

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

FROM base as take_home_project
COPY . /usr/src/app
WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get install ca-certificates -y

ENTRYPOINT [ "python" ]
CMD [ "views.py" ]