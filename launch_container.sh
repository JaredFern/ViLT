# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='2'
fi


docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=/data/jaredfer/vilt,dst=/data,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /src liangkeg/vilt:v1
