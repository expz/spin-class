#!/bin/bash

. ./venv/bin/activate

export LD_LIBRARY_PATH="$HOME/.mujoco/mjpro150/bin:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

jupyter lab \
  --ip='127.0.0.1' \
  --port=8884 \
  --no-browser \
  --notebook-dir=~/Dropbox/devbox/src/spinning-up
