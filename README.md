# ExactUnlearn

ExactUnlearn is a project focused on unlearning in recommendation systems. It consists of two main frameworks:

**RaCoMU Framework**: A solution for unlearning in recommendation systems that removes the influence of specific training data while preserving model performance.

**CCGU Framework**: A graph-based precise unlearning framework designed to efficiently remove sensitive data by leveraging the structural information of graph-based recommendation systems.

# Dataset
Movielens-1m:https://grouplens.org/datasets/movielens/1m/

Course Recommendation:http://moocdata.cn/data/course-recommendation

Movielesn-100k：https://grouplens.org/datasets/movielens/100k/

Amazon-Bookshttps://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data

# Reproduce Experiments


# Environment Setup

**Hardware Environment**: 

RTX 4060(8G)

**Software Environment:**

RaCoMU:

python3.6

jupyter-client 7.1.2

pip install torch==1.10.2+cu113

CCGU:
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

pip install torch-geometric
