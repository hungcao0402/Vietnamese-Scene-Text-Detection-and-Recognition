FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV PATH="/root/anaconda3/bin:$PATH" FORCE_CUDA="1" \
    TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
    
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update --allow-releaseinfo-change && apt-get install -y libgbm-dev -y \
    software-properties-common dirmngr -y \
    build-essential -y \
    libgl1-mesa-glx libxrender1 libfontconfig1 -y \
    libglib2.0-0 -y \
    libsm6 libxext6 libxrender-dev -y \
    vim zip unzip wget screen -y \
    gnupg2 -y \
    libgl1-mesa-glx -y \
    git libmagickwand-dev -y

RUN cd /root && wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh \
    && cd /root && bash Anaconda3-2020.07-Linux-x86_64.sh -b -p ./anaconda3 \
    && bash -c "source /root/anaconda3/etc/profile.d/conda.sh"

# RUN cd /root && wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh

# RUN cd /root && bash Anaconda3-2020.07-Linux-x86_64.sh -b -p ./anaconda3

# RUN bash -c "source /root/anaconda3/etc/profile.d/conda.sh && conda install -y pytorch==1.5.0 torchvision cudatoolkit=10.2 -c pytorch"

# RUN bash -c "/root/anaconda3/bin/conda init bash"
WORKDIR /root
RUN mkdir code && mkdir data
WORKDIR code

RUN git clone https://github.com/facebookresearch/detectron2.git
RUN git clone https://github.com/aim-uofa/AdelaiDet.git adet
# RUN bash -c "source /root/anaconda3/etc/profile.d/conda.sh && conda activate base && cd detectron2 && python setup.py build develop"
RUN conda init bash \
    && . ~/.bashrc \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install ninja yacs \
                cython matplotlib tqdm opencv-python shapely \
                scipy tensorboardX pyclipper Polygon3 \
                weighted-levenshtein editdistance easydict \
    && python3 -m pip install --upgrade Pillow \
    && python3 -m pip install dict_trie
RUN python3 -m pip install torch==1.5.1 torchvision==0.6.1 gdown

RUN python3 -m pip install git+https://github.com/facebookresearch/detectron2.git@9eb4831f742ae6a13b8edb61d07b619392fb6543
    # && conda install -y pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch \
# RUN bash -c "cd detectron2 && python3 setup.py build develop"

WORKDIR /
RUN mkdir bkai
WORKDIR /bkai

# RUN rm /root/Anaconda3-2020.07-Linux-x86_64.sh
