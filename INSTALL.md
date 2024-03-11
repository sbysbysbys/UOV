# Installation
**Step 1.** Install [PyTorch and Torchvision](https://pytorch.org/get-started/previous-versions/)

```shell
conda create -n SSOV python=3.9
conda activate SSOV
# we use torch.__version__='1.11.0+cu113'
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
``` 

**Step 2.** Install [Detection2](https://github.com/pytorch/vision/):
```
# python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
# pip install git+https://github.com/cocodataset/panopticapi.git
```

**Step 3.** Install MinkowskiEngine:
```
conda install openblas-devel -c anaconda
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
pip install ninja
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```


**Step 3.** For more detailed information, please refer to [FC-CLIP_INSTALL](https://github.com/bytedance/fc-clip/blob/main/INSTALL.md), [CAT-Seg_INSTALL](https://github.com/KU-CVLAB/CAT-Seg/blob/main/INSTALL.md), [SAN_INSTALL](https://github.com/MendelXu/SAN).
```
conda install pytorch-scatter -c pyg
pip install -r requirements.txt
# for fc-clip
pip install git+https://github.com/mlfoundations/open_clip.git --prefer-binary
cd ov_segment/fc-clip/fcclip/modeling/pixel_decoder/ops
sh make.sh
```