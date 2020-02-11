## [Ultra-high Resolution EM Images Segmentation Challenge](https://www.biendata.com/competition/urisc/)

#### [Final Rank](https://www.biendata.com/competition/urisc/winners): 2nd in simple track, 5th in complex track

### Environment
#### Hardware

- 4x Tesla V100 (GPU Memory 32GB each)
- CPU Memory 250GB

#### Packages
> pip install -r requirements.txt

### Data
Processed data can be downloaded [here](https://pan.baidu.com/s/1LrP56-fstinTh3cNUtTRKg). Put it in top-level folder.

### Model
#### Pretrained Model:
ResNet-50: [Download](https://hangzh.s3.amazonaws.com/encoding/models/resnet50-25c4b509.zip), 
ResNet-101: [Download](https://hangzh.s3.amazonaws.com/encoding/models/resnet101-2a57e44d.zip),
ResNet-152: [Download](https://hangzh.s3.amazonaws.com/encoding/models/resnet152-0d43d698.zip)

#### Simple Track
[DFF](https://arxiv.org/abs/1902.09104), backbone ResNet-50

#### Complex Track
[CASENet](https://arxiv.org/abs/1705.09759), backbone ResNet-152


### Training

#### Simple Track
> CUDA\_VISIBLE\_DEVICES=0,1,2,3 python train.py --dataset simple --aug --k 1

#### Complex Track
> CUDA\_VISIBLE\_DEVICES=0,1,2,3 python train.py --dataset complex --aug --kernel-size 9 --edge-weight 0.4


### Testing and Ensembling
#### Simple Track
> CUDA\_VISIBLE\_DEVICES=0 python test.py --dataset simple --model DFF --backbone resnet50

#### Complex Track
> CUDA\_VISIBLE\_DEVICES=0 python test.py --dataset complex --model CASENet --backbone resnet152
