# 🚌 NYC MTA Transport Delay Analysis

## Business Problem
NYC MTA had no predictive visibility into which routes,
stops, and times cause maximum delays — costing 
commuters millions of hours annually.

## Solution
Built end-to-end transport analytics pipeline:
- Analyzed 5M+ real MTA bus trip records
- Identified worst routes, stops, and time windows
- Built ML model to predict delays before they happen

## Key Findings
- 🔴 Friday 6PM = worst delay window (13.8 min avg)
- 🚌 Route S86 = worst performer (12.75 min avg delay)
- 🛑 N.J. Turnpike stop = 80+ min delays at peak
- ⏰ Hour of day + Route = top delay predictors (65% importance)
- 🤖 ML model predicts delays with MAE of ~4 minutes

## Tech Stack
| Tool         | Purpose                              |
|--------------|--------------------------------------|
| Python       | Data cleaning, ML model              |
| Pandas       | Data manipulation (5M rows)          |
| SQLAlchemy   | Python to MySQL connection           |
| MySQL        | Store & query 5M trip records        |
| Scikit-learn | Random Forest delay predictor        |
| Matplotlib   | Feature importance visualization     |
| Tableau      | Interactive 4-page dashboard         |

## Project Structure
transport-delay-project/
├── data/
│   ├── raw/
│   │   └── mta_1712.csv
│   └── processed/
│       ├── delay_overview.csv
│       ├── route_performance.csv
│       ├── peak_analysis.csv
│       ├── stop_analysis.csv
│       ├── feature_importance.csv
│       ├── feature_importance.png
│       └── model_predictions.csv
├── python/
│   ├── load_data.py
│   ├── run_queries.py
│   └── delay_predictor.py
├── sql/
│   └── queries.sql
└── README.md

## How to Run
1. pip install pandas sqlalchemy pymysql scikit-learn matplotlib seaborn
2. python python/load_data.py
3. python python/run_queries.py
4. python python/delay_predictor.py
5. Open Tableau → connect CSVs from data/processed/

## Dataset
- Source: MTA New York City (Kaggle)
- Records: 4,965,896 trips
- Period: December 2017
- 17 original features → 7 engineered features
