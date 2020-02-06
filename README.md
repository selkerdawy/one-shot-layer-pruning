# One-Shot Layer-Wise Accuracy Approximation for Layer Pruning

This is a demo for the proposed method on VGG19_bn cifar100 to generate Figure 3 in the paper.

## Setup requirements
```bash
# Virtual environment creation
virtualenv .envpy36 -p python3.6
source .envpy36/bin/activate
#Install libraries
pip install -r req.txt
```
## pretrained weights
Download pretrained weights for CIFAR100 vgg19_bn from [here](https://drive.google.com/file/d/1Lj6XmhG7TtNnQ5I0bVxItXlnRsnDopf6/view?usp=sharing)

## Run 
```bash
python imprint_cifar.py -d cifar100 --arch vgg19_bn --pretrained PATH_TO_MODEL -c cifar100_vgg
```


