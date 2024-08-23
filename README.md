# deeplearningbasics

## GCP VM Instance NVDIA Machine Setup

```
sudo apt-get install lsof
sudo apt-get install -y poppler-utils
sudo apt install python3
sudo apt-get install docker-compose-plugin


    1  conda info
    2  conda create -n llm python=3.11
    3  conda activate llm
# export DRIVER_VERSION=535.183.01
sudo apt update
sudo apt --list upgradable
sudo apt upgrade
sudo apt autoremove nvidia* --purge
sudo apt-get remove --purge '^nvidia-.*'
sudo /usr/bin/nvidia-uninstall 
nvida-smi
nvcc
sudo add-apt-repository contrib
lscu|grep CPU
lscpu|grep CPU
sudo apt install linux-headers-amd64 
sudo apt install nvidia-detect
sudo apt install nvidia-driver linux-image-amd64
#sudo /opt/deeplearning/install-driver.sh 
 
    4  nvidia-smi
    5  conda install pytorch torchvision torchaudio cudatoolkit -c pytorch-nightly
or
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
conda install nvidia/label/cuda-12.0.0::cuda-toolkit
    6  nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
    7 pip install ipython
    8 python version : Python 3.11.9
    9 pip install ipykernel
   10 ipython kernel install --user --name=llm
   11 python -m ipykernel install --user --name=llm
   12 pip install jupyter
   13 jupyter notebook --generate-config
   14 update config
   c.NotebookApp.ip = '*'
   c.NotebookApp.open_browser = False
   c.NotebookApp.port = 8083
   15 jupyter notebook password
   16 nohup jupyter notebook --no-browser --port 8083

!pip install torch
Collecting torch
  Using cached torch-2.3.1-cp311-cp311-manylinux1_x86_64.whl.metadata (26 kB)
Collecting filelock (from torch)
  Using cached filelock-3.15.1-py3-none-any.whl.metadata (2.8 kB)
Requirement already satisfied: typing-extensions>=4.8.0 in /home/sridhanya_ganapathi_team_neustar/llm/lib/python3.11/site-packages (from torch) (4.12.2)
Collecting sympy (from torch)
  Using cached sympy-1.12.1-py3-none-any.whl.metadata (12 kB)
Collecting networkx (from torch)
  Using cached networkx-3.3-py3-none-any.whl.metadata (5.1 kB)
Requirement already satisfied: jinja2 in /home/sridhanya_ganapathi_team_neustar/llm/lib/python3.11/site-packages (from torch) (3.1.4)
Collecting fsspec (from torch)
  Using cached fsspec-2024.6.0-py3-none-any.whl.metadata (11 kB)
Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch)
  Using cached nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-cuda-runtime-cu12==12.1.105 (from torch)
  Using cached nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-cuda-cupti-cu12==12.1.105 (from torch)
  Using cached nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cudnn-cu12==8.9.2.26 (from torch)
  Using cached nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cublas-cu12==12.1.3.1 (from torch)
  Using cached nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-cufft-cu12==11.0.2.54 (from torch)
  Using cached nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-curand-cu12==10.3.2.106 (from torch)
  Using cached nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-cusolver-cu12==11.4.5.107 (from torch)
  Using cached nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cusparse-cu12==12.1.0.106 (from torch)
  Using cached nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-nccl-cu12==2.20.5 (from torch)
  Using cached nvidia_nccl_cu12-2.20.5-py3-none-manylinux2014_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-nvtx-cu12==12.1.105 (from torch)
  Using cached nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.7 kB)
Collecting triton==2.3.1 (from torch)
  Using cached triton-2.3.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.4 kB)
Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch)
  Using cached nvidia_nvjitlink_cu12-12.5.40-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
Requirement already satisfied: MarkupSafe>=2.0 in /home/sridhanya_ganapathi_team_neustar/llm/lib/python3.11/site-packages (from jinja2->torch) (2.1.5)
Collecting mpmath<1.4.0,>=1.1.0 (from sympy->torch)
  Using cached mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Using cached torch-2.3.1-cp311-cp311-manylinux1_x86_64.whl (779.2 MB)
Using cached nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
Using cached nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
Using cached nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
Using cached nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
Using cached nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
Using cached nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
Downloading nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.5/56.5 MB 33.3 MB/s eta 0:00:0000:0100:01
Downloading nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 124.2/124.2 MB 17.0 MB/s eta 0:00:0000:0100:01
Downloading nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 196.0/196.0 MB 4.2 MB/s eta 0:00:0000:0100:01m
Downloading nvidia_nccl_cu12-2.20.5-py3-none-manylinux2014_x86_64.whl (176.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 176.2/176.2 MB 3.6 MB/s eta 0:00:0000:0100:02
Downloading nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 99.1/99.1 kB 4.6 MB/s eta 0:00:00
Downloading triton-2.3.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (168.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 168.1/168.1 MB 10.1 MB/s eta 0:00:0000:0100:01
Using cached filelock-3.15.1-py3-none-any.whl (15 kB)
Downloading fsspec-2024.6.0-py3-none-any.whl (176 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 176.9/176.9 kB 22.6 MB/s eta 0:00:00
Downloading networkx-3.3-py3-none-any.whl (1.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 25.2 MB/s eta 0:00:00a 0:00:01
Downloading sympy-1.12.1-py3-none-any.whl (5.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.7/5.7 MB 13.6 MB/s eta 0:00:0000:0100:01
Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 kB 26.9 MB/s eta 0:00:00
Downloading nvidia_nvjitlink_cu12-12.5.40-py3-none-manylinux2014_x86_64.whl (21.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 21.3/21.3 MB 26.2 MB/s eta 0:00:0000:0100:01
Installing collected packages: mpmath, sympy, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, networkx, fsspec, filelock, triton, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, torch
Successfully installed filelock-3.15.1 fsspec-2024.6.0 mpmath-1.3.0 networkx-3.3 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.20.5 nvidia-nvjitlink-cu12-12.5.40 nvidia-nvtx-cu12-12.1.105 sympy-1.12.1 torch-2.3.1 triton-2.3.1

import torch
# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. PyTorch was installed with CUDA support.")
    # Print CUDA device information
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. PyTorch was installed without CUDA support.")


For Open CV with cuda support
https://gist.github.com/minhhieutruong0705/8f0ec70c400420e0007c15c98510f133
https://forum.opencv.org/t/can-i-use-opencv-python-with-gpu/8947/2

milvus db setup
==============

wget https://github.com/milvus-io/milvus/releases/download/v2.3.18/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d
For Attu
docker run -p 8084:3000 -e HOST_URL=http://10.113.8.40:8084 -e MILVUS_URL=<host> zilliz/attu:latest

To Enable autocomplete in jupyternotebook
https://stackoverflow.com/questions/76893872/modulenotfounderror-no-module-named-notebook-base-when-installing-nbextension

!pip3 install jupyter-tabnine
!jupyter nbextension install --py jupyter_tabnine
!jupyter nbextension enable --py jupyter_tabnine
!jupyter serverextension enable --py jupyter_tabnine

jupyter contrib nbextension install --user
   85  pip install --upgrade notebook==6.4.12
   86  pip uninstall traitlets
   87  pip install traitlets==5.9.0
   88  pip install --upgrade notebook==6.4.12 traitlets==5.9.0 jupyter jupyter_contrib_nbextensions
   89  conda install -c conda-forge jupyter_contrib_nbextensions
   90  conda install -c "conda-forge/label/cf201901" jupyter_contrib_nbextensions
   91  conda install -c "conda-forge/label/cf202003" jupyter_contrib_nbextensions
   92  jupyter nbextension enable codefolding/main
   93  jupyter nbextension enable --py jupyter_tabnine
   94  pip3 install jupyter-tabnine
   95  upyter nbextension enable --py jupyter_tabnine
   96  jupyter nbextension enable --py jupyter_tabnine
   97  pip3 install jupyter-tabnine

base) sridhanya_ganapathi@sridhanya-neo4j:~/neo4j$ cat docker-compose.yml 
version: '3.3'
services:
  neo4j:
    image: neo4j:latest
    restart: always
    container_name: my_neo4j_container
    ports:
      - "8081:7474"  # Map container's 7474 (Neo4j web port) to host's 8084
      - "8080:7687"  # Map container's 7687 (Bolt port) to host's 7687
    environment:
      NEO4J_AUTH: neo4j/password  # Set initial username and password
      NEO4J_ACCEPT_LICENSE_AGREEMENT: "yes"
      NEO4J_dbms_security_procedures_unrestricted: gds.*,apoc.*
    volumes:
      - ./conf:/conf
      - ./data:/data
      - ./import:/import
      - ./logs:/logs
      - ./plugins:/plugins

https://gist.github.com/MihailCosmin/affa6b1b71b43787e9228c25fe15aeba
 vi /opt/deeplearning/driver-version.sh
export DRIVER_VERSION=535.183.01

wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-debian11-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-debian11-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-debian11-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda

export PATH=$PATH:/usr/local/cuda-12.2/bin
## For GPU Support
sudo apt install -y nvidia-docker2
sudo systemctl daemon-reload
sudo systemctl restart docker


import os
os.environ['USER_AGENT'] = 'myagent'

289  sudo apt update

  290  sudo apt --list upgradable

  291  sudo apt upgrade

  292  sudo apt autoremove nvidia* --purge

  293  sudo /usr/bin/nvidia-uninstall 

  294  clear

  295  nvida-smi

  296  nvcc

  297  sudo add-apt-repository contrib

  298  lscu|grep CPU

  299  lscpu|grep CPU
 
export DRIVER_VERSION=535.183.01
 
 
sudo /opt/deeplearning/install-driver.sh
 




```
