
# WSI Colon Polyp Subtype Classification Using Graph Neural Networks
[Final Report.pdf](https://github.com/maedemir/EECE-571F-Project/files/13994767/Final.Report.pdf)


# Introduction
In this project, our goal is to represent Gigantic Whole Slide Images (WSI) of colon polyp tissues as graphs and apply graph neural networks for subtype classification.
![](Pipeline.png)


# Getting Started
The code has several steps to do classification on Whole Slide Images (WSI). The steps are:

1. Applying [HoverNet](https://github.com/vqdang/hover_net) on patches to extract nuclei
2. Based on HoverNet's output (step 2), extract images of each nuclei
3. Based on step 3's output, extract feature for each nuclei by using deep neural network encoder (such as swin or resnet18) (Assume features are saved in X variable)
4. Based on HoverNet's output (step 2), generate a knn graph for each patch (assume adjacency matrix is A)
5. Based on step 4 and 5, we have graphs and features for each node (A and X). Each of these patches have a label (subtype). We train a Graph Neural Network (GNN) based model on data.

## Run
First of all clone the repository:

``` bash
git clone https://github.com/alibalapour/EECE-571F-Project.git
```

In the next step, install requirements by running this command:

``` bash
pip install -r EECE-571F-Project/requirements.txt
```

## Steps
### 1. Run HoverNet
Once you have the project set up and the requirements installed, you can use HoverNet for nucleus detection. Run these commands:

``` bash
code/scripts/extract_nuclei_[CLS].sh '[image directory]' '[output directory]'
```

Note that you need to select CLS from 'HP', 'SSL', 'TA', and 'NCM'.


### 2. Run Extract Cell Images bash
To extract cell images, for each class (CLS) run below command:

``` bash
code/scripts/extract_cells_[CLS].sh
```

*Note:* You need to set the input and output path inside the bash.


### 3. Run Extract Features of each Images cell
To extract features of cells in a patch based on each class (CLS) run this command:

``` bash
code/scripts/extract_cell_features_[CLS].sh
```

*Note:* You need to set the input and output path inside the bash.

### 4. Run Extract Graph
To extract graphs of each patch, run this command:

``` bash
code/scripts/extract_graph_[CLS].sh
```

*Note:* You need to set the input and output path inside the bash.


### 5. Run Classification 
To train and test a GNN-based model on the generated graphs and features, run below command:

``` bash
python 'code/classification.py' --graph_pairs_path=[path_to_graph_pairs] --features_dir_path=[path_to_features]
```




