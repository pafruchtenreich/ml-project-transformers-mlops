FROM ubuntu:22.04

# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip

WORKDIR /code

# Install project dependencies
COPY requirements.txt .
COPY main.py .
COPY src/ ./src
RUN pip install -r requirements.txt

RUN chmod +x src/app/run.sh

CMD ["bash","src/app/run.sh"]
