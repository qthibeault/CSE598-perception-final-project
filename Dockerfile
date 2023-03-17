from ros:humble

ARG NVIM_TAG=stable

RUN apt-get update
RUN apt-get install -y make gettext libtool-bin cmake g++ pkg-config unzip curl git python3-pip

RUN git clone --depth=1 --branch=$NVIM_TAG https://github.com/neovim/neovim /tmp/neovim
RUN make -C /tmp/neovim CMAKE_BUILD_TYPE=Release
RUN make -C /tmp/neovim install
RUN rm -rf /tmp/neovim

RUN pip install setuptools==58.2.0 pynvim

RUN curl -fsSL https://deb.nodesource.com/setup_current.x | sudo -E bash -
RUN apt-get install -y nodejs

RUN mkdir ~/.config
RUN git clone https://gitlab.com/qthibeault/nvim ~/.config/nvim
