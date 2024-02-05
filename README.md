# dycomed

Python scripts to perform eddy detections, tracking and collocation with remote-sensing and in situ data, in delayed time or near-real-time mode

# Getting started

Install the python packages and dependencies, with  mamba to go faster (and attempting directly flexible solve):
```
conda env create -n dycoenv
conda activate dycoenv
conda install conda-libmamba-solver mamba --freeze-installed
mamba env update -f dyco_environment.yml
```
In this environment install py-eddy-tracker :
```
git clone https://github.com/AntSimi/py-eddy-tracker
python setup.py install
```
In the folder with you codes create a file `secretcodes.py` with only your CMEMS credentials :
```
cmems_username='xxxxxx'
cmems_password='xxxxxx'
```

## Downloading data from CMEMS server

- run `0_refresh_SSH_NRT.py` to download NRT SSH L4 data
- run `0_refresh_SST_chloro.py` to download NRT SST or Chlorophyll L3S data
- run `2_ftp_latest.py` to download in-situ data.

Codes are similar but for SSH a `refresh` option is available to update the previous files at day+3 or +6. 

## Performing NRT eddy detections

run `1_pet_obs_refresh.py` to compute eddy detections from Py-Eddy-Tracker.
They will be stored in an common Atlas at the end



