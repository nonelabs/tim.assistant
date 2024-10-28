#!/bin/bash
if [ ! -e venv/bin/activate ];then python -m venv venv;fi
source venv/bin/activate
CMAKE_ARGS="-DGGML_CUDA=ON -DLLAVA_BUILD=OFF" pip install llama-cpp-python --force --no-cache-dir
pip install -r requirements.txt