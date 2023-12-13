# EECE-571F-Project

# Introduction

# Getting Started
The code has several steps to do classification on Whole Slide Images (WSI). The steps are:
1. Extracting patches from WSI
2. Applying [HoverNet](https://github.com/vqdang/hover_net) on patches to extract nuclei
3. Based on HoverNet's output (step 2), extract images of each nuclei
4. Based on step 3's output, extract feature for each nuclei by using deep neural network encoder (such as swin or resnet18) (Assume features are saved in X variable)
5. Based on HoverNet's output (step 2), generate a knn graph for each patch (assume adjacency matrix is A)
6. Based on step 4 and 5, we have graphs and features for each node (A and X). Each of these patches have a label (subtype). We train a Graph Neural Network (GNN) based model on data.

## Run
First of all clone the repository:
```
git clone https://github.com/alibalapour/EECE-571F-Project.git
```

In the next step, install requirements by running this command:
```
pip install -r EECE-571F-Project/requirements.txt
```

## Run HoverNet
Once you have the project set up and the requirements installed, you can use HoverNet for nucleus detection. Follow these steps:

### 1. Prepare Your Data:
Make sure your images are located in a directory, which we will refer to as `$img_dir`. You should also specify an output directory, which we will refer to as `$output_dir`.

### 2. Run the HoverNet Script:
Run the HoverNet script by executing the following command. This script will extract nuclei from your images and save the results in the specified output directory.
```
chmod +x EECE-571F-Project/code/extract_nuclei.sh
EECE-571F-Project/code/extract_nuclei.sh '$img_dir' '$output_dir'
```


