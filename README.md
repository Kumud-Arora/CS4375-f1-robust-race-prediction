# CS4375-f1-robust-race-prediction
# F1 Race Outcome Prediction
 
Predicts Formula 1 podium finishes using Gradient Boosting implemented from scratch.
 
**Course:** CS 4375 — Introduction to Machine Learning  
**Dataset:** [Formula 1 World Championship 1950-2024](https://www.kaggle.com/datasets/muhammadehsan02/formula-1-world-championship-history-1950-2024)
 
## Project structure
 
```
f1-ml-project/
  data/               ← place all Kaggle CSV files here
  src/
    data_loader.py        ← loads and cleans CSVs
    feature_engineering.py← builds model features
    eda.py                ← exploratory analysis + plots
    gradient_boosting.py  ← gradient boosting coded from scratch
    train_evaluate.py     ← training, evaluation, adversarial analysis
  outputs/
    figures/          ← all plots saved here
    experiment_log.csv← auto-generated experiment log
  requirements.txt
  README.md
```
 
## Setup
 
```bash
python -m venv venv
source venv/bin/activate        # For Windows use: venv\Scripts\activate
pip install -r requirements.txt
```
 
## Download data
 
1. Go to the Kaggle link above
2. Download all CSVs
3. Place them in the `data/` folder
## Run
 
```bash
cd src
python data_loader.py       # verify data loads correctly
python eda.py               # generate EDA plots
python train_evaluate.py    # train model + all evaluation
```
 
## Results
 
See `outputs/experiment_log.csv` for a log of all experiments and metrics.
