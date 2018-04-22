# Mood Forecasting Using Sensory Data Obtained Using Smart-phones


This project using a time series of sensory data and mood ratings to predict what mood someone will have the next day.
 

## File structure
- src folder
	- main.py: main script
	- util.py: functions used by all models 
	- benchmark.py: benchmark model 
	- ml.py: lightgbm model
	- temporal.py: ARIMA model
	- arima\_tuning.py: optimize perameters for ARIMA model
- output folder
	- arima\_best\_cfg.pkl: optimized perameters for ARIMA models
	- pred\_arima.csv: prediction file from ARIMA model
	- pred\_benchmark.csv: prediction file from benchmark model
	- pred\_ml.csv: prediction file from lightgbm model
- data file is not included 

## Pipeline
1. Obtain the data
2. Extract information from the dataset
3. Processing dataset
4. Split training and test set
4. Using three different models to predict mood.
5. Evaluation

## How to run the scripts
``` sh
python main.py -m benchmark -o output/pred_benchmark.csv  -e output/score_benchmark.csv
python main.py -m ml -o output/pred_ml.csv  -e output/score_ml.csv
python main.py -m temporal_algorithm -o output/pred_arima.csv  -e output/score_arima.csv
```