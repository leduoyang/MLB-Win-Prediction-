# MLB-Win-Prediction
Predict MLB wins per season based on past data by regression model with some analyses
source : https://www.datacamp.com/community/tutorials/scikit-learn-tutorial-baseball-1

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

## Importing data
Import data from database to df(DataFrame) with pandas
* Each row of our data will consist of a single team for a specific year
data from https://github.com/jknecht/baseball-archive-sqlite

```
+ Read data by quering a sqlitedatabase using the sqlite3 package
+ Convert th fetch data to a DataFrame with pandas
the data will be filtered to only include currently active teams and only years where the team played 150 or more games
```

## Data pre-processing
Add column headers to dataframe , remove columns which aren't associated to the target column 'W' and deal with null values

### Add column headers
Passing a list of headers to the columns attribeute from pandas to add headers 
```
# names for the columns of dataframe
cols = ['yearID','lgID','teamID','franchID','divID','Rank','G','Ghome','W','L','DivWin','WCWin','LgWin','WSWin','R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','name','park','attendance','BPF','PPF','teamIDBR','teamIDlahman45','teamIDretro','franchID','franchName','active','NAassoc']

teams_df.columns = cols
```
### Remove not necessary columns with respect to 'Win'
```
# Dropping unnecesary column variables.
drop_cols = ['lgID','franchID','divID','Rank','Ghome','L','DivWin','WCWin','LgWin','WSWin','SF','name','park','attendance','BPF','PPF','teamIDBR','teamIDlahman45','teamIDretro','franchID','franchName','active','NAassoc']

df = teams_df.drop(drop_cols, axis=1)
```

### Deal with null values

#### Display the count of null values for each column
```
# Print out null values of all columns of `df`
print(df.isnull().sum(axis=0).tolist())
```

#### set SO(110 null values) and DP(22) to median of the column 
(because the amount of null values to both columns are relatively small)
```
filling the null values by the fillna() method

e.g.
df['SO'] = df['SO'].fillna(df['SO'].median())
df['DP'] = df['DP'].fillna(df['DP'].median())
```
#### remove/drop CS(419 null values) and HBP(1777) columns
(Two of the columns have relatively high amount of null values)
```
df = df.drop(['CS','HBP'], axis=1)
```

## Exploring and Visualizing the data
+ histogram of the win distribution
+ scatter plot to show the quantized data according to # of wins
+ figure about average runs per-game with respect to years

### histogram of the win distribution with matplotlib
```
e.g.
plt.hist(df['W'])
plt.xlabel('Wins')
plt.title('Distribution of Wins')

plt.show()
```

### scatter plot to show the quantized data according to # of wins
create a new column win_bins by apply() method on the column 'W' and function 'assign_win_bins()'
assign_win_bins : quantized the column 'W' from integer value to 1~5 bins
```
df['win_bins'] = df['W'].apply(assign_win_bins)
```

after extracting the column 'win_bins', make a scatter graph :
```
plt.scatter(df['yearID'], df['W'], c=df['win_bins'])
plt.title('Wins Scatter Plot')
plt.xlabel('Year')
plt.ylabel('Wins')

plt.show()
```

we can see in the above scatter plot, there are very few seasons from before 1900
because of that, it makes sense to eliminate those rows from data sets
```
df = df[ df['yearID'] > 1900 ] 
```

### figure about average runs per-game with years to indicate how much scoring there was for each year
As MLB progressed, different eras emerged where the amount of runs per game increased or decreased significantly. The dead ball era of the early 1900s is an example of a low scoring era and the steroid era at the turn of the 21st century is an example of a high scoring era.

steps:
1. create dictionaries runs-per-year and games-per-year by using the 'iterrows()' method to loop through dataframe
2. create a dictionary called mlb-runs-per-game by iterate through the games-per-year with 'items()' method
3. sort mlb-runs-per-game and save as list befor unzip it

plot from the mlb-runs-per-game dictionary :
```
plt.plot(x, y)
plt.title('MLB Yearly Runs per Game')
plt.xlabel('Year')
plt.ylabel('MLB Runs per Game')

plt.show()
```









