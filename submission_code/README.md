# Text Classification
Project2 of EPFL Machine Learning Course by by R. Urbanke and M. Jaggi

##Team
- Nie Shiyue
- Xu Jiacheng

##Code Structure
- The `*.sh` files are used for building vocabluary
- The method used for preprocessing, training model, prediction and submission are found in `utilities.py`
- The `data` folder contains the original full training data and test data. The data must should be put in this folder for exectuting `run.py`

##Additional Libraries
- pandas
- sklearn
- nltk
- numpy

##Reproduce Submission
- This code can fix on the linux OS (As the bash scripts be added as the subprocess of `run.py`,and running on the windows may need to change the path )
- For reproducing best submission on Kaggle,simply run:
```
$ python3 run.py
```