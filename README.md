## Ultra-high Resolution EM Images Segmentation Challenge

**2nd place in the Ultra-high Resolution EM Images Segmentation [Challenge](https://www.biendata.xyz/competition/urisc/) hosted by [BAAI](https://www.baai.ac.cn/) and [PKU](http://english.pku.edu.cn/)**

## Environment
#### Hardware

- 4 NVIDIA Tesla V100 GPUs (32GB memory each)
- CPU memory 250GB

#### Packages
```bash
pip install -r requirements.txt
```

## Data
Processed data can be downloaded [here](https://pan.baidu.com/s/1LrP56-fstinTh3cNUtTRKg). Put it in top-level folder.

## Model
#### Pretrained Model:
ResNet-50: [Download](https://hangzh.s3.amazonaws.com/encoding/models/resnet50-25c4b509.zip), 
ResNet-101: [Download](https://hangzh.s3.amazonaws.com/encoding/models/resnet101-2a57e44d.zip),
ResNet-152: [Download](https://hangzh.s3.amazonaws.com/encoding/models/resnet152-0d43d698.zip)

#### Simple Track
[DFF](https://arxiv.org/abs/1902.09104), backbone ResNet-50

#### Complex Track
[CASENet](https://arxiv.org/abs/1705.09759), backbone ResNet-152


## Training

#### Simple Track
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset simple --model DFF --backbone resnet50 --batch-size 4 --lr 0.0014 --epochs 200 --crop-size 960 --aug --k 1
```

#### Complex Track
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset complex --model CASENet --backbone resnet152 --batch-size 4 --lr 0.0014 --epochs 45 --crop-size 1280 --aug --kernel-size 9 --edge-weight 0.4
```

## Testing and Ensembling
#### Simple Track
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset simple --model DFF --backbone resnet50
```
#### Complex Track
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset complex --model CASENet --backbone resnet152
```
