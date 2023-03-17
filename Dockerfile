from ros:humble

# Add nodesource deb repository
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | sudo -E bash -

# Install image dependencies
RUN apt-get update
RUN apt-get install -y make \
    gettext \
    libtool-bin \
    cmake \
    g++ \
    pkg-config \
    unzip \
    curl \
    git \
    python3-pip

# Neovim version to install
ARG NVIM_TAG=stable

# Build and install Neovim from source
RUN git clone --depth=1 --branch=$NVIM_TAG https://github.com/neovim/neovim /tmp/neovim
RUN make -C /tmp/neovim CMAKE_BUILD_TYPE=Release
RUN make -C /tmp/neovim install
RUN rm -rf /tmp/neovim

# Pin setuptools dependency to avoid deprecation warning for `setup.py install`
RUN pip install setuptools==58.2.0

# Install NodeJS to support Pyright language server
RUN apt-get install -y nodejs

# Install python development tools
RUN pip install ruff black isort

# Clone neovim configuration
RUN git clone https://gitlab.com/qthibeault/nvim ~/.config/nvim
