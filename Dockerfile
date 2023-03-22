from ros:humble

# Update apt sources
RUN apt-get update

# Install image dependencies
RUN apt-get install -y \
    build-essential \
    gettext \
    libtool-bin \
    cmake \
    pkg-config \
    unzip \
    curl \
    git \
    python3-pip

# Add nodesource deb repository
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash -

# Install nodejs from nodesource repository
RUN apt-get install -y nodejs

# Pin setuptools dependency to avoid deprecation warning for `setup.py install`
RUN pip install setuptools==58.2.0

# Install colcon dependencies
RUN --mount=type=bind,target=/workspace \
    rosdep install --from-paths /workspace/src -y --ignore-src

# Neovim version to install
ARG NVIM_TAG=stable

# Build and install Neovim from source
RUN git clone --depth=1 --branch=$NVIM_TAG https://github.com/neovim/neovim /tmp/neovim \
    && make -C /tmp/neovim CMAKE_BUILD_TYPE=Release \
    && make -C /tmp/neovim install \
    && rm -rf /tmp/neovim

# Install python development tools
RUN pip install ruff black isort

# Clone neovim configuration
RUN git clone https://gitlab.com/qthibeault/nvim /root/.config/nvim

# Pre-install neovim plugins
RUN nvim --headless "+Lazy! install" +qa


RUN cat <<EOF >> /root/.bashrc
source /opt/ros/humble/setup.bash

if [ -d "/workspace/install" ]; then
    source /workspace/install/setup.bash
fi
EOF
