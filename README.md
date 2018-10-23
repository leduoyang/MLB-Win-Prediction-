# MLB-Win-Prediction
Predict win per year based on past data by regression model with some analyses

## Prerequisites
* scikit-learn
* pandas
* sqlite3
* matplotlib


## Workflow
+ Import data from database to df(DataFrame) with pandas
+ data pre-processing - add column headers to df, dealing with null values 
+ exploring and visualizing the data - distribution of wins by hist, show scatter plot from quantized data , figure about runs per-game with years
+ feature extracting - one-hot encoding(get_dummies) according to yearID, runs per-game,runs allowed per-game
+ Build the predicition model - add clustering information by K-means as feature, split data to train and test , build model by regression , select MAE(mean absolute error ) as error metric






