# ExactUnlearn

ExactUnlearn is a project focused on unlearning in recommendation systems. It consists of two main frameworks:

**RaCoMU Framework**: A solution for unlearning in recommendation systems that removes the influence of specific training data while preserving model performance.

**CCGU Framework**: A graph-based precise unlearning framework designed to efficiently remove sensitive data by leveraging the structural information of graph-based recommendation systems.

# Dataset
Movielens-1m:https://grouplens.org/datasets/movielens/1m/

Course Recommendation:http://moocdata.cn/data/course-recommendation

Movielesn-100k：https://grouplens.org/datasets/movielens/100k/

Amazon-Bookshttps://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data

# Directory Structure
- `/CCGU/`: Contains the code and resources for the "Contrastive Clustering Graph Unlearning" model.
- `/RaCo/`: Contains code for the "Rapid Collaborative Machine Unlearning" project.

# Environment Setup

**Hardware Environment**: 

12th Gen Intel(R)Core(TM)i5-12400F2.50 GHz

NVIDIA GeForce RTX 4060 (8 GB)

**Software Environment:**

**RaCoMU:**
- **Python**: 3.6
- **PyTorch**: 
  - `torch==1.10.2+cu113`
- **Jupyter**: 
  - `jupyter-client==7.1.2`
  - `jupyterlab==3.2.1`
  - `jupyter-server==1.4.1`
- **Other dependencies**:
  - `numpy==1.19.5`
  - `scikit-learn==0.24.2`
  - `matplotlib==3.3.4`
  - `pandas==1.1.5`
  - `scipy==1.4.1`
  - `requests==2.27.1`
  - `pyyaml==6.0.1`

**CCGU:**
- **Python**: 3.6
- **PyTorch**:
  - `torch==1.10.2+cu113` 
  
- **Jupyter**:
  - `jupyter-client==7.1.2`
  - `jupyterlab==3.2.1`
  - `jupyter-server==1.4.1`
  
- **Other Dependencies**:
  - `numpy==1.19.5`
  - `scikit-learn==0.24.2`
  - `matplotlib==3.3.4`
  - `pandas==1.1.5`
  - `scipy==1.4.1`
  - `requests==2.27.1`
  - `pyyaml==6.0.1`



Install PyTorch Geometric Dependencies:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
Install PyTorch Geometric:
```bash
pip install torch-geometric
```
