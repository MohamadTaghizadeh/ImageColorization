# Use Python 3.12.0 as base image
FROM python:3.12.0

ENV LANG=C.UTF-8
ARG PIP_TIMEOUT=500



# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN python -m pip install --upgrade pip

# Copy requirements first for better caching
ADD ./requirements.txt /srv/
# Install Python dependencies
RUN python -m pip install -r /srv/requirements.txt --timeout $PIP_TIMEOUT


# Expose the port the app runs on
WORKDIR /approot/src
