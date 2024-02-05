# dycomed

Python scripts to perform eddy detections, tracking and collocation with remote-sensing and in situ data

### Getting started

Install the python packages and dependencies :
```
conda env create -n dycoenv -f environment.yml
```
Install mamba to go faster :
conda install conda-libmamba-solver --freeze-installed
```
conda activate dycoenv
```
In this environment install py-eddy-tracker :
```
git clone https://github.com/AntSimi/py-eddy-tracker
python setup.py install
```

### Downloading data from CMEMS server

In the folder with you codes create a file `secretcodes.py` with only your CMEMS credentials :
```
cmems_username='xxxxxx'
cmems_password='xxxxxx'
```
run 
