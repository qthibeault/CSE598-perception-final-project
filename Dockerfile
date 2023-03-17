from ubuntu:jammy

ARG NVIM_TAG=stable

RUN apt-get update
RUN apt-get install -y make gettext libtool-bin cmake g++ pkg-config unzip curl git

RUN git clone --depth=1 --branch=$NVIM_TAG https://github.com/neovim/neovim /tmp/neovim
RUN make -C /tmp/neovim CMAKE_BUILD_TYPE=Release

FROM ros:humble

COPY --from=0 /tmp/neovim /tmp/neovim
RUN make -C /tmp/neovim install
RUN rm -r /tmp/neovim

RUN apt-get update
RUN apt-get install -y python3-pip git
RUN pip install setuptools==58.2.0 pynvim

RUN mkdir ~/.config
RUN git clone https://gitlab.com/qthibeault/nvim ~/.config/nvim
