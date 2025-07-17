# SweUniAdmissionsAnalysis
Scripts for making plots of data from Swedish national admissions systems. 

# How to install
The scripts use standard packages like `numpy` and `matplotlib`, and some for making maps use `swemaps` and `geopandas`. The needed packages are listed in `requirements.txt`, and can be installed via `pip install -r requirements.txt`

So, e.g. on a Mac you can do something like this to set up a virtual python environment and install the dependencies
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Data
A few different sources provide useful info that is then read by the scripts in this repo:
* The admissions data portal NyA provides Excel spreadsheets with info about the applicants in each admission round, with names, addresses and merit scores for the applicants.
* The Swedish Council for Higher Education, [UHR](https://www.uhr.se/en/start/), publish summary statistics about the number of applicants, grades/scores needed to get into educational programs, etc. These also come as Excel files

# Available scripts
A couple of different scripts are foreseen for making plots using data from a few different sources. Right now we only have one to give geographical overview of applicant and admission data.
* `analyze_geo.py`

