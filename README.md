# CSE598 Perception in Robotics - Final Project

The objective of this project is to track an object and predict its trajectory
through an occlusion.

## Prerequisites

This project requires [Docker](https://docs.docker.com/engine/install) and
[Just](https://just.systems/man/en/chapter_4.html).

## Setup

The environment for this project is defined in the `Dockerfile` at the root of
this repository. The easiest way to create this environment is to run the
command `just build-image`.

## Building

To build all of the [ROS](https://ros.org) modules for this project, you can run
the command `just build-modules`.

## Modules

### Detector

The detector module is responsible for identifying the object of interest in the
image and correlating the object between images. The detector uses a Mask R-CNN
defined in the [PyTorch](https://pytorch.org) library for detection.

The detector module can be started in a container using the command `just
run-module detector`.

### Predictor

The predictor module is responsible for taking position of the object of
interest and estimating its trajectory through space.

The predictor module can be started in a container using the command `just
run-module predictor`.

## References

TODO
