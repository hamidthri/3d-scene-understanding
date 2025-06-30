# PointNet-Custom
A custom reimplementation of the PointNet architecture for point cloud classification using the ModelNet10 dataset.


## Architecture
![PointNet Architecture](https://github.com/hamidthri/pointnet-custom/blob/master/assets/pointnet.png)

## Prerequisites

Install dependencies:
```bash
pip install requirements.txt
```

## Installation
Clone the repository:
```bash
git clone https://github.com/hamidthri/pointnet-custom.git
cd pointnet-custom
```

## Dataset
Download and extract the ModelNet10 dataset:
```bash
wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip ModelNet10.zip
```


## Usage

Train Model:
```bash
python train.py
```

Evaluate Model:
```bash
python predict.py
```




