# PointNet-Custom
A custom reimplementation of the PointNet architecture for point cloud classification using the ModelNet10 dataset.


## Architecture
![PointNet Architecture](images/pointnet.png)

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

## Results

The PointNet-Custom model was evaluated on the **ModelNet10** test set, consisting of 908 samples across 10 object categories. The classification report below summarizes precision, recall, and F1-score per class:

### Classification Report

| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Bathtub     | 0.98      | 0.88   | 0.93     | 50      |
| Bed         | 0.97      | 0.95   | 0.96     | 100     |
| Chair       | 0.99      | 0.99   | 0.99     | 100     |
| Desk        | 0.62      | 0.88   | 0.73     | 86      |
| Dresser     | 0.50      | 0.34   | 0.40     | 86      |
| Monitor     | 0.99      | 0.96   | 0.97     | 100     |
| Night Stand | 0.89      | 0.48   | 0.62     | 86      |
| Sofa        | 0.95      | 0.99   | 0.97     | 100     |
| Table       | 0.90      | 0.65   | 0.76     | 100     |
| Toilet      | 0.60      | 0.99   | 0.75     | 100     |

- **Overall accuracy:** 82%  
- **Macro average F1-score:** 81%

The model performs very well on classes like **chair**, **bed**, and **monitor**, while performance on **dresser** and **night_stand** is lower, likely due to fewer distinctive features or less training data.

### Example Prediction

Below is a visualization of a sample point cloud from the test set alongside its predicted class label:

![Sample Prediction](assets/prediction_sample.png)

*Visualization shows the point cloud colored by predicted class confidence.*



