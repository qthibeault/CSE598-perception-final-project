FROM ros:humble

ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID ros
RUN useradd --uid $USER_UID --gid $USER_GID -M ros
RUN apt-get update
RUN apt-get install -y sudo
RUN echo ros ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/ros
RUN chmod 0440 /etc/sudoers.d/ros

USER ros
