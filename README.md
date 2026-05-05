# 🏎️ F1 Race Outcome Prediction

### Robust Prediction of Formula 1 Race Outcomes Using Gradient Boosting with Adversarial Evaluation and Decision Simulation

This project builds a from-scratch Gradient Boosting model to predict Formula 1 podium finishes using historical race data from 1950–2024. It was developed as part of my Introduction to Machine Learning (CS 4375) course at UT Dallas, with a focus on implementing the core algorithm manually to better understand how it works under the hood, instead of relying on libraries like sklearn.

## ⚙️ Methodological Background

Gradient Boosting, introduced by Friedman, builds models iteratively by fitting each new learner to the residual errors (negative gradients) of the previous one. With small updates at each step, it forms a strong, well-regularized ensemble that performs especially well on noisy, real-world data.

> *Friedman showed that boosting reduces error iteratively; Formula 1 shows that error can evolve.*

## Dataset: [Formula 1 World Championship 1950-2024](https://www.kaggle.com/datasets/muhammadehsan02/formula-1-world-championship-history-1950-2024)

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
python3 -m venv venv
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
python3 data_loader.py       # verify data loads correctly
python3 eda.py               # generate EDA plots
python3 train_evaluate.py    # train model + all evaluation
```
 
## Results
 
See `outputs/experiment_log.csv` for a log of all experiments and metrics.
