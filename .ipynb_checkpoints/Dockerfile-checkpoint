# Use a Python base image
FROM python:3.7-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the source code and configuration file into the container
COPY src/ /app/src/
COPY config/ /app/config/
COPY pipeline_component.py /app/
COPY requirements.txt /app/

# Install any dependencies your code requires
RUN \
apt-get update && \
apt-get upgrade -y && \
apt-get autoremove -y && \
apt-get clean -y && \
pip install --upgrade pip && \
pip install wheel && \
pip install -r /app/requirements.txt

# Set environment variables
ENV CONFIG_PATH /app/config/config.yaml

# Define the command to run your pipeline component
CMD ["python", "pipeline_component.py"]