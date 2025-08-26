# Use Python 3.12.0 as base image
FROM python:3.12.0

WORKDIR /approot/src
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
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

# Install Python dependencies one by one to better handle errors
RUN pip install --no-cache-dir numpy==2.3.0 && \
    pip install --no-cache-dir opencv-python==4.11.0.86 && \
    pip install --no-cache-dir torch==2.7.1 && \
    pip install --no-cache-dir torchvision==0.22.1 && \
    pip install --no-cache-dir ffmpeg-python==0.2.0 && \
    pip install --no-cache-dir pyyaml==6.0.2 



# Copy current directory contents into container
ADD ./requirements.txt /srv/
RUN python -m pip install -r /srv/requirements.txt --timeout $PIP_TIMEOUT



# Set the line endings to LF
RUN find . -type f -name "*.py" -exec sed -i 's/\r$//' {} +
