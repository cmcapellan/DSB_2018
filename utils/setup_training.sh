#!/bin/bash
apt-get update && apt-get install -y python3-pip kaggle
pip3 install gpustat
git clone https://github.com/cmcapellan/DSB_2018.git
mkdir /root/.kaggle && scp root@52.117.88.215:/root/.kaggle/kaggle.json /root/.kaggle/kaggle.json
kaggle competitions download -c data-science-bowl-2018 -p /root
unzip /root/data-science-bowl-2018.zip && mkdir /root/stage1_train && unzip /root/stage1_train.zip -d /root/stage1_train
docker build --rm -t tf -f ~/DSB_2018/utils/Dockerfile.nucleii .
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 /root
nvidia-docker run -it -v /root:/root -p 6006:6006 -w /root tf python3 -u /root/DSB_2018/mrcnn/my_train_1.py