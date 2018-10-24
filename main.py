"""
Scikit-Learn Tutorial: Baseball Analytics Pt 1
source : https://www.datacamp.com/community/tutorials/scikit-learn-tutorial-baseball-1
"""

# import `pandas` and `sqlite3`
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression , RidgeCV
from sklearn.metrics import mean_absolute_error
################### function ################### 

# Creating bins for the win column
def assign_win_bins(W):
    if W < 50:
        return 1
    if W >= 50 and W <= 69:
        return 2
    if W >= 70 and W <= 89:
        return 3
    if W >= 90 and W <= 109:
        return 4
    if W >= 110:
        return 5
def assign_era_label(year):    
    if year < 1920:
        return 1
    elif year >= 1920 and year <= 1941:
        return 2
    elif year >= 1942 and year <= 1945:
        return 3
    elif year >= 1946 and year <= 1962:
        return 4
    elif year >= 1963 and year <= 1976:
        return 5
    elif year >= 1977 and year <= 1992:
        return 6
    elif year >= 1993 and year <= 2009:
        return 7
    elif year >= 2010:
        return 8    
def assign_mlb_rpg(year):
    return avg_run_per_game[year]
################### fetching data from database ################### 


# Connecting to SQLite Database
conn = sqlite3.connect('lahman2016.sqlite')

# Querying Database for all seasons where a team played 150 or more games and is still active today. 
query = ''' select * from Teams inner join TeamsFranchises on Teams.franchID == TeamsFranchises.franchID
where Teams.G >=150 and TeamsFranchises.active =='Y'; '''

# Creating dataframe from query.
Teams = conn.execute(query).fetchall()
# Convert `Teams` to DataFrame
teams_df = pd.DataFrame(Teams)

#print(teams_df.head(5))

################### Cleaning and Preparing The Data ################### 

# Adding column names to dataframe
cols = ['yearID','lgID','teamID','franchID','divID','Rank','G','Ghome','W','L','DivWin',
        'WCWin','LgWin','WSWin','R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF',
        'RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','name',
        'park','attendance','BPF','PPF','teamIDBR','teamIDlahman45','teamIDretro','franchID',
        'franchName','active','NAassoc']

teams_df.columns =  cols

# Eliminating the columns that aren't necessary or derived from the target column(Wins)
drop_cols = ['lgID','franchID','divID','Rank','Ghome','L','DivWin',
             'WCWin','LgWin','WSWin','SF','name','park','attendance',
             'BPF','PPF','teamIDBR','teamIDlahman45','teamIDretro',
             'franchID','franchName','active','NAassoc']

df = teams_df.drop(drop_cols,axis=1)

# Dealing with null values of columns
isnull = df.isnull().sum(axis=0).tolist()
print(isnull)

# Eliminating columns with large amount of null values
df = df.drop(['CS','HBP'],axis=1)

# Filling null values with the median values of the columns
df['SO'] = df['SO'].fillna(df['SO'].median())
df['DP'] = df['DP'].fillna(df['DP'].median())

isnull = df.isnull().sum(axis=0).tolist()
print(isnull)

################### Exploring and Visualizing The Data ################### 
plt.hist(df['W'])
plt.xlabel('Wins')
plt.title('Distribution of Wins')
plt.show()

print("Average wins per year :")
print(df['W'].mean())

# Create bins of wins by applying `assign_win_bins` to `df['W']`    
df['win_bins'] = df['W'].apply(assign_win_bins)

# Plotting scatter graph of Year vs. Wins
plt.scatter(df['yearID'], df['W'], c=df['win_bins'])
plt.title('Wins Scatter Plot')
plt.xlabel('Year')
plt.ylabel('Wins')

plt.show()


# figure of average runs-per-game with respect to years
#runs_per_year , games_per_year = getRuns_Games_per_year(df)

runs_per_year , games_per_year = {},{}
for i,row in df.iterrows(): #Iterate over DataFrame rows as (index, Series) pairs.
    year, runs, games = row['yearID'],row['R'],row['G']
    if year in runs_per_year:
        runs_per_year[year] = runs_per_year[year] + runs
        games_per_year[year] = games_per_year[year] + games        
    else:
        runs_per_year[year] = runs
        games_per_year[year] = games  
avg_run_per_game = {}
for year,runs in runs_per_year.items():
    games = games_per_year[year]
    avg_run_per_game[year] = runs / games

lists = sorted(avg_run_per_game.items())
# .items() return a iterable object and sorted() can sort any iterable object
x , y = zip(*lists) #zip(*list) for unzip

plt.plot(x, y)
plt.title('MLB Yearly Runs per Game')
plt.xlabel('Year')
plt.ylabel('MLB Runs per Game')

plt.show()


################### adding feature ################### 
#era with one-hot encoding
df['era'] = df['yearID'].apply(assign_era_label)
dummy_df = pd.get_dummies(df['era'],prefix='era')

df = pd.concat([df,dummy_df],axis=1)

#mlb average runs per game derived above
df['mlb_rpg'] = df['yearID'].apply(assign_mlb_rpg)

#run per game
df['R_per_game'] = df['R'] / df['G']
#run allowed per game
df['RA_per_game'] =  df['RA'] / df['G']
#show scatter plot
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.scatter(df['R_per_game'], df['W'], c='blue')
ax1.set_title('Runs per Game vs. Wins')
ax1.set_xlabel('Runs per Game')
ax1.set_ylabel('Wins')
ax2.scatter(df['RA_per_game'], df['W'], c='red')
ax2.set_title('Runs Allowed per Game vs. Wins')
ax2.set_xlabel('Runs Allowed per Game')
ax2.set_ylabel('Wins')

plt.show()

# how each variables/features is correlated with target variable
C = df.corr()['W']

# set K-means to features
# create a new dataframe and set columns needed for clustering
attributes = ['G','R','AB','H','2B','3B','HR','BB','SO','SB','RA','ER','ERA','CG',
'SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','era_1','era_2','era_3','era_4','era_5','era_6','era_7','era_8','R_per_game','RA_per_game','mlb_rpg']

data_attributes = df[attributes]

kmeans = KMeans(n_clusters=6, random_state=1)
distances = kmeans.fit_transform(data_attributes)
labels = kmeans.labels_

plt.scatter(distances[:,0], distances[:,1], c=labels)
plt.title('Kmeans Clusters')
plt.show()

df['labels'] = labels
attributes.append('labels')





################### buliding model ################### 
numeric_cols = ['G','R','AB','H','2B','3B','HR','BB','SO','SB','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','era_1','era_2','era_3','era_4','era_5','era_6','era_7','era_8','R_per_game','RA_per_game','mlb_rpg','labels','W']
data = df[numeric_cols]



train = data.sample(frac=0.75, random_state=1)
test = data.loc[~data.index.isin(train.index)]

x_train = train[attributes]
y_train = train['W']
x_test = test[attributes]
y_test = test['W']



lr = LinearRegression(normalize = True)
lr.fit(x_train,y_train)
predictions = lr.predict(x_test)

mae = mean_absolute_error(y_test,predictions)
print(mae)


rrm =  RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0), normalize=True)
rrm.fit(x_train,y_train)
predictions = rrm.predict(x_test)

mae = mean_absolute_error(y_test,predictions)
print(mae)






