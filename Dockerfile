FROM ros:humble
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip install setuptools==58.2.0
