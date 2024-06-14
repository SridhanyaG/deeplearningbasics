# deeplearningbasics

## GCP VM Instance NVDIA Machine Setup

```
sudo apt-get install lsof

    1  conda info
    2  conda create -n llm
    3  conda activate llm
    4  nvidia-smi
    5  conda install pytorch torchvision torchaudio cudatoolkit -c pytorch-nightly
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
```
