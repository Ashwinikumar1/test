FROM gcr.io/spark-operator/spark-py:v3.0.0
MAINTAINER Engage

# using root user
USER root:root

# create directory for apps
RUN mkdir -p /app


# copy spark program
COPY test.py /app/

# set work directory
WORKDIR /app

# user
USER 1001
