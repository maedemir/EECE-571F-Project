# EECE-571F-Project

### Getting Started
First of all clone the repository:
```
git clone https://github.com/alibalapour/EECE-571F-Project.git
```

In the next step, install requirements by running this command:
```
pip install -r EECE-571F-Project/requirements.txt
```

### Run HoverNet
Once you have the project set up and the requirements installed, you can use HoverNet for nucleus detection. Follow these steps:

**1. Prepare Your Data:**
Make sure your images are located in a directory, which we will refer to as $img_dir. You should also specify an output directory, which we will refer to as $output_dir.

**2. Run the HoverNet Script:**
Run the HoverNet script by executing the following command. This script will extract nuclei from your images and save the results in the specified output directory.
```
chmod +x EECE-571F-Project/code/extract_nuclei.sh
EECE-571F-Project/code/extract_nuclei.sh '$img_dir' '$output_dir'
```


