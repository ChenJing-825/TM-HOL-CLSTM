# TM-HOL: Topic Memory model for Detection of Hate Speech and Offensive Language

## Requirements
* TensorFlow >= 1.2.1
* Keras >= 2.0.8
* gensim >= 2.1.0

## Embedding
http://nlp.stanford.edu/projects/glove/
## Input data format
A sampled data is provided in `data/tmn/labeled_data.csv` 

## Prerequisites:
Required packages are listed in the requirements.txt file:
```
pip install -r requirements.txt
```
## How to run
Preprocess data:
```
python process_tmn.py 
```
Run TMN:
```
python tmn_run.py    
```

More detailed configurations can be found in `tmn_run.py`.

## Disclaimer
The code is for research purpose only and released under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
