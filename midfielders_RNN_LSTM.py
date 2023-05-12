import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, recall_score, \
    precision_score, f1_score
from yellowbrick.cluster import KElbowVisualizer
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Functions that return the birth year of each player, an empty/not empty results, finally a season value
def birthyear(string):
    return int(string.split('-')[0])

def isempty(a):
    return a != a

def season(x):
    s1 = [11, 35, 43, 52, 40]
    s3 = [5, 33, 41, 50, 38]
    s4 = [150, 151, 152, 153, 154]

    if x['competitionSeasonId'] in s1:
        val = 1
    elif x['competitionSeasonId'] in s3:
        val = 3
    elif x['competitionSeasonId'] in s4:
        val = 4
    else:
        val = 2
    return val

# Function that drops highly correlated variables
def hcorr_drop(df):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    drop_p = [column for column in upper.columns if any(upper[column] > 0.75)]
    drop_n = [column for column in upper.columns if any(upper[column] < -0.75)]
    df.drop(drop_p, axis=1, inplace=True)
    df.drop(drop_n, axis=1, inplace=True)
    return df

# PLAYERS ARE ALREADY FILTERED BY A MINIMUM OF 450 MINUTES per SEASON
player_stats_old = pd.read_csv('SeasonPlayerStats_18_19.csv', sep=';')
player_comps_old = pd.read_csv('CompetitionPlayers_18_19.csv', sep=';')
player_comps_old.rename(columns={'id': 'playerId'}, inplace=True)
player_roles_old = pd.read_csv('PlayerRoles_18_19.csv', sep=';')
player_stats_new = pd.read_csv('SeasonPlayerStats_10comp.csv', sep=';')
player_roles_new = pd.read_csv('PlayerRoles_10comp.csv', sep=';')
player_dict = pd.read_csv('PlayerDict.csv', sep=';')
player_dict.rename(columns={'kamaId': 'playerId'}, inplace=True)
player_comps_new = pd.read_csv('CompSeasPlayers_10comp.csv', sep=';', encoding='unicode_escape')
player_comps_new.rename(columns={'id': 'playerId'}, inplace=True)
# parameter_details = pd.read_csv('ParameterDetails.csv', sep=';', encoding='unicode_escape')

player_stats = pd.concat([player_stats_old, player_stats_new])
seasons = [11, 35, 43, 52, 40, 5, 33, 41, 50, 38, 150, 151, 152, 153, 154, 1, 34, 39, 42, 51]
player_stats = player_stats[player_stats['competitionSeasonId'].isin(seasons)]
player_comps = pd.concat([player_comps_new, player_comps_old])
player_roles = pd.concat([player_roles_new, player_roles_old]).drop_duplicates(subset=['playerId'], keep='first')

# Handling and fixing player valuations dataset (scraped)
transfermarkt = pd.read_csv('player_valuations.csv')
transfermarkt.drop(['player_club_domestic_competition_id', 'current_club_id', 'dateweek', 'datetime'],
                   axis=1, inplace=True)
transfermarkt['date'] = pd.to_datetime(transfermarkt['date'], format="%d/%m/%Y", dayfirst=True)
transfermarkt['year'] = transfermarkt['date'].dt.year
y = [2018, 2019, 2020, 2021, 2022]
transfermarkt = transfermarkt[transfermarkt['year'].isin(y)]

# season 18-19 ---> 15/07/2018 to 30/05/2019
# season 19-20 ---> 15/07/2019 to 30/05/2020
# season 20-21 ---> 15/07/2020 to 30/05/2021
# season 21-22 ---> 15/072021 to 30/05/2022
mask1 = (transfermarkt['date'] > '2018-07-30') & (transfermarkt['date'] <= '2019-05-30')
mask2 = (transfermarkt['date'] > '2019-07-30') & (transfermarkt['date'] <= '2020-05-30')
mask3 = (transfermarkt['date'] > '2020-07-30') & (transfermarkt['date'] <= '2021-05-30')
mask4 = (transfermarkt['date'] > '2021-07-30') & (transfermarkt['date'] <= '2022-05-30')

a = transfermarkt.loc[mask1]
a.pop('year')
a['season'] = 1
b = transfermarkt.loc[mask2]
b.pop('year')
b['season'] = 2
c = transfermarkt.loc[mask3]
c.pop('year')
c['season'] = 3
d = transfermarkt.loc[mask4]
d.pop('year')
d['season'] = 4
players0 = pd.concat([a, b, c, d])
players0.pop('date')
players_trns = players0.groupby(['transfermarktId', 'season'], as_index=False).mean()


# grouping minutes by comps, players
tab = player_stats[['playerId', 'minutes', 'competitionSeasonId', 'teamId']]
tab = tab.groupby(['playerId', 'competitionSeasonId', 'teamId']).mean().reset_index()

# pivoting parameters p90
new_stats = pd.pivot_table(player_stats, index=['playerId', 'competitionSeasonId'],
                           columns=['parameterCode'], values='P90').reset_index()

# merging mkt value, role, birthdate
df = new_stats.merge(player_comps, on='playerId').merge(player_dict, on='playerId').merge(
    player_roles, on='playerId').drop_duplicates()

# merging to have total minutes
df = df.merge(tab, on=['playerId', 'competitionSeasonId'])

df['season'] = df.apply(season, axis=1)

year = []
for i in df['birthDate']:
    x = birthyear(i)
    year.append(x)

df['birth'] = year
df.pop('birthDate')
df.pop('marketValue')

df_final = df.merge(players_trns, how='left', on=['transfermarktId', 'season'])

seriea = [1, 11, 154, 5]
premleague = [34, 35, 150, 33]
bundes = [50, 51, 52, 151]
liga = [38, 39, 40, 153]
competition = []
for x in df_final['competitionSeasonId'].values:
    if x in seriea:
        competition.append('Serie A')
    elif x in premleague:
        competition.append('Premier League')
    elif x in bundes:
        competition.append('Bundesliga')
    elif x in liga:
        competition.append('La Liga')
    else:
        competition.append('Ligue One')

df_final['Competition'] = competition
df_final.drop(['roleId', 'transfermarktId', 'name', 'role', 'shortName_en', 'competitionSeasonId',
               'teamId'], axis=1, inplace=True)
df_final.sort_values(by='playerId', inplace=True)

print(df_final['marketValue'].isna().sum())  # counting NAs within market value variable
df_final = df_final[df_final['marketValue'].notna()]  # removing rows with NA market value (416 observations)

# Extract goalkeepers from the dataset and then removing gks' related variables
goalkeepers = df_final[df_final['instatPositions'] == 31]
df_final = df_final[df_final['instatPositions'] != 31]
df_final.drop(['GOL-ALLS', 'GOL-BS', 'GOL-OBS', 'GOL-S', 'PAS-M', 'PRT', 'PRT-X', 'PRT-SUP', 'RIG-P',
               'TIS', 'TIS-S', 'TIS-B', 'TIS-OB', 'TIS-SB', 'TIS-SOB', 'USC'], axis=1, inplace=True)

storage = df_final[['playerId', 'foot', 'height', 'weight', 'season']]
nas_tot = df_final.isna().sum()  # checking for nas all across the dataset
df_final.drop(['ACC', 'TIR-FS', 'RIG-RX', 'CRS-R', 'GOL-PD', 'PAR-L30', 'GOL-ALL'],
              axis=1, inplace=True)  #variables with huge number of nas + DECIDE WHAT TO DO WITH LAST THREE VARS

df_final.drop(['CRS', 'DRB', 'DUL-AD', 'DUL-AO', 'DUL-AV', 'DUL', 'DUL-A', 'DUL-D', 'DUL-O', 'DUL-T', 'DUL-TV',
               'DUL-V', 'FAL-D', 'FAL-O', 'GOL-NPX', 'FAL-R', 'GOL-A', 'GOL-NP', 'GOL-RP', 'GOL-R', 'OCG-OB', 'OCG-B',
               'OCG-BF', 'PAR-CG', 'RIG-R', 'PAS-CG', 'PAS-COR', 'PAS-DD', 'PAS-FT', 'PAS-BCKRRX', 'PAS-BCK', 'PAS-FTR',
               'PAS-FWD', 'PAS-FWDRRX', 'PAS-LO', 'PAS-L', 'PAS-O', 'PLG-AA', 'TIR-NP', 'TIR-SB', 'TIR-SOB', 'TIR-SNP',
               'TIR-W', 'TIR-T', 'foot', 'height', 'weight'], axis=1, inplace=True)
#df_final.set_index('playerId', inplace=True)

## EXPLORATORY DATA ANALYSIS ##

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

#birth
#pt = df_final.pivot_table(index='birth', values='playerId', aggfunc='nunique')
#ax = pt.plot(kind='bar', rot=0, legend=False)
#plt.xlabel('Players Birthyear', fontsize=16)
#plt.xticks(rotation=45)
#plt.ylabel('Frequency', fontsize=16)
#plt.title('Year of birth distribution', fontsize=18)
#plt.show()

#pt2 = df_final.pivot_table(index='birth', columns='Competition', values='playerId', aggfunc='nunique')
#ax2 = pt2.plot(kind='bar', rot=0)
#plt.xlabel('Year of Birth', fontsize=16)
#plt.xticks(rotation=45)
#plt.ylabel('Frequency', fontsize=16)
#plt.legend(fontsize=13)
#plt.title('Year of birth distribution across competitions', fontsize=18)
#plt.show()


#------BOXPLOT MARKET VALUE vs BIRTH-------------
#sns.boxplot(x=df_final['birth'], y=df_final['marketValue']/1000000)
#plt.xlabel('Year of Birth', fontsize=14)
#plt.ylabel('Market Value in millions', fontsize=14)
#plt.title('Market Value vs Birth', fontsize=16)
#plt.show()

#------BOXPLOT MARKET VALUE vs COMPETITION-------------
#sns.boxplot(x=df_final['Competition'], y=df_final['marketValue']/1000000)
#plt.xlabel('Competition', fontsize=16)
#plt.ylabel('Market Value in millions of €', fontsize=16)
#plt.title('Market Value vs Competition', fontsize=18)
#plt.show()


#total market value across 4 years through different leagues
seriea_tot = []
pl_tot = []
bundes_tot = []
liga_tot = []
ligue1_tot = []
years = [2018, 2019, 2020, 2021]
for x in range(1, 5):
    seriea_tot.append(
        df_final.loc[(df_final['Competition'] == 'Serie A') & (df_final['season'] == x), 'marketValue'].sum())
    pl_tot.append(
        df_final.loc[(df_final['Competition'] == 'Premier League') & (df_final['season'] == x), 'marketValue'].sum())
    bundes_tot.append(
        df_final.loc[(df_final['Competition'] == 'Bundesliga') & (df_final['season'] == x), 'marketValue'].sum())
    liga_tot.append(
        df_final.loc[(df_final['Competition'] == 'La Liga') & (df_final['season'] == x), 'marketValue'].sum())
    ligue1_tot.append(
        df_final.loc[(df_final['Competition'] == 'Ligue One') & (df_final['season'] == x), 'marketValue'].sum())

data = {"Serie A": seriea_tot, "Premier League": pl_tot, "Bundesliga": bundes_tot, "La Liga": liga_tot,
            "Ligue One": ligue1_tot, "Years": years}
totals = pd.DataFrame(data)


# sns.lineplot(x=totals['Years'], y=totals['Serie A']/1000000000, estimator=None, linewidth=3, label='Serie A')
# sns.lineplot(x=totals['Years'], y=totals['Premier League']/1000000000, estimator=None, linewidth=3, label='Premier League')
# sns.lineplot(x=totals['Years'], y=totals['Bundesliga']/1000000000, estimator=None, linewidth=3, label='Bundesliga')
# sns.lineplot(x=totals['Years'], y=totals['La Liga']/1000000000, estimator=None, linewidth=3, label='La Liga')
# sns.lineplot(x=totals['Years'], y=totals['Ligue One']/1000000000, estimator=None, linewidth=3, label='Ligue One')
# plt.title('Total Market Value Trend across the different leagues', fontsize=18)
# plt.xlabel('Years', fontsize=16)
# plt.ylabel('Total Market Value in € (billions)', fontsize=16)
# plt.legend(loc='lower left', fontsize=13)
# plt.xticks(totals['Years'])
# plt.yticks(range(0, 12, 1))
# plt.show()

# scatter minutes-mkt value
# min_vdm= df_final.groupby(['playerId']).agg({'minutes':np.mean, 'marketValue':np.mean})
# sns.scatterplot(x=min_vdm['minutes'], y=min_vdm['marketValue']/1000000)
# plt.title('Total Minutes - Market Value Scatterplot', fontsize=16)
# plt.xlabel('Total Minutes', fontsize=14)
# plt.ylabel('Market Value (millions)', fontsize=14)
# plt.show()

# DISTRIBUTION PLOT for VDM
# sns.displot(x=df_final['marketValue']/1000000, kde=True, palette=sns.color_palette('tab10'))
# plt.title('Market Value Distribution Curve', fontsize=18)
# plt.xlabel('Market Value in millions of €', fontsize=16)
# plt.ylabel('Frequency', fontsize=16)
# plt.show()

# dropping vdm greater then a certain threshold (100 millions)
df_final = df_final[df_final.marketValue <= 80000000]

# desc = df_final[['minutes', 'GOL', 'PAS', 'PLG', 'DUL-VX', 'ASS-V', 'marketValue']].describe(include='all')
# desc.to_csv('summary_statistics.csv')


#CREATION OF 'age' VARIABLE
years = [2019, 2020, 2021, 2022]
def subtract_from_year(row):
    return years[row['season']-1] - row['birth']
df_final['age'] = df.apply(subtract_from_year, axis=1)


# MANIPULATION OF 'age' VARIABLE
df_final['age^2'] = df_final['age'] ** 2
df_final.pop('birth')


# MANIPULATION OF 'marketValue' VARIABLE
# df_final['ln_mktval'] = np.log(df_final['marketValue'])
# sns.displot(x=df_final['ln_mktval'], kde=True)
# plt.title('Market Value Distribution', fontsize=18)
# plt.xlabel('Natural logarithm of Market Value', fontsize=16)
# plt.ylabel('Count', fontsize=16)
# plt.show()


transformed_vars = df_final[['playerId', 'season', 'marketValue', 'age']]
df_final.drop(['age'], axis=1, inplace=True)


# Instat Positions in order to create sub-samples for each macro-role
d = [12, 13, 22, 32, 42, 52, 53]
m = [23, 33, 43, 14, 24, 34, 44, 54, 35, 25, 45]
f = [15, 16, 26, 36, 46, 55, 56]


# NA imputer [KNN imputer from sklearn]
imputer = KNNImputer(n_neighbors=25, missing_values=np.nan)
competition_storage = df_final[['playerId', 'season', 'Competition']]
df_final.pop('Competition')
df_final = pd.DataFrame(imputer.fit_transform(df_final), columns=df_final.columns)

zero_df = (df_final == 0).astype(int).sum(axis=0)
df_final.drop(['CRT-R', 'GOL-C', 'GOL-OB', 'GOL-P', 'GOL-PI', 'GOL-T'], axis=1, inplace=True)  # too many ZEROs
df_final = df_final.merge(competition_storage, on=['playerId', 'season'])


# CREATE RANK VARIABLE FOR COMPETITION
mapping = {'Premier League': 5, 'La Liga': 4, 'Serie A': 3, 'Bundesliga': 2, 'Ligue One': 1}
df_final['comp_rank'] = df_final['Competition'].map(mapping)
df_final.pop('Competition')


# MIDFIELDERS
midfielders = df_final[df_final['instatPositions'].isin(m)]
midfielders.pop('instatPositions')

# ECDF for midfielders
# sns.ecdfplot(x=midfielders['marketValue']/1000000)
# plt.title('Midfielders Market Value ECDF Curve', fontsize=18)
# plt.xlabel('Market Value in millions of €', fontsize=16)
# plt.ylabel('Proportion', fontsize=16)
# plt.show()

hcorr_drop(midfielders)
# removing non-relevant variables
midfielders.drop(['DRB-RX', 'PAS-BCKRX'], axis=1, inplace=True)
midfielders.sort_values(['playerId'], ascending=[True], inplace=True)
# mid_params = list(midfielders.columns)
midfielders.set_index('playerId', inplace=True)

# midfielders binning
bins_mid = [0, 2500000, 5000000, 8000000, 12000000, 15000000, 20000000, 25000000,
            30000000, 40000000, 50000000, 80000000]
labels_mid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
midfielders_bins = pd.cut(midfielders['marketValue'], bins=bins_mid, labels=labels_mid, precision=0)
mid_bins = midfielders_bins.value_counts()


mid_desc = midfielders.describe()

def detect_outliers(df):
    outliers = {}
    for col in df.columns:
        # Calculate the Z-score of each value in the column
        z = np.abs(stats.zscore(df[col]))
        # Identify outliers as values with a Z-score greater than 3
        outliers[col] = np.sum(z > 3)
    return outliers


mid_outliers = detect_outliers(midfielders)
mid_describe = midfielders.describe()

midfielders.reset_index(drop=True, inplace=True)
midfielders_bins.reset_index(drop=True, inplace=True)
midfielders['vdm_binned'] = midfielders_bins
vdm_midfielders = midfielders.pop('marketValue')
y_mid = midfielders.pop('vdm_binned')


# balancing through SMOTE algorithm
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(midfielders, y_mid)  # oversampling

df = X_resampled.copy()
df['target'] = y_resampled
X = df.loc[:, df.columns != 'target'].to_numpy()
yValues = df.loc[:, 'target'].to_numpy()
y = to_categorical(yValues)

# df = midfielders.copy()
# df['target'] = y_mid
# X = df.loc[:, df.columns != 'target'].to_numpy()
# yValues = df.loc[:, 'target'].to_numpy()
# y = to_categorical(yValues)

# dataframe split [the commented line of code is the one that doesn't use SMOTE and produces worst results]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler_mid = StandardScaler()
X_train = scaler_mid.fit_transform(X_train)
X_test = scaler_mid.transform(X_test)
# midfielders_scaled = pd.DataFrame(scaler_mid.fit_transform(midfielders), columns=midfielders.columns)

# callback
patience = 5
callback = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

classificator = Sequential()
classificator.add(LSTM(128, activation='tanh', input_shape=(X.shape[1], 1), return_sequences=True))
classificator.add(LSTM(128, activation='tanh'))
classificator.add(Dense(y.shape[1], activation='softmax'))
classificator.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classificator.summary()

#  plot_model(classificator, show_shapes=True, show_layer_names=True)

hist_def = classificator.fit(X_train, y_train, epochs=200, verbose=1, validation_data=(X_test, y_test),
                             callbacks=[callback])

y_pred = classificator.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
print("Accuracy score:", accuracy_score(y_test_classes, y_pred_classes))
print("")
print("Recall score:", recall_score(y_test_classes, y_pred_classes, average='weighted'))
print("")
print("F1 score:", f1_score(y_test_classes, y_pred_classes, average='weighted'))
print("")
print("Precision score:", precision_score(y_test_classes, y_pred_classes, average='weighted'))
print("")
# print(classification_report(y_test_classes, y_pred_classes))

# Confusion matrix plot
# cm = confusion_matrix(y_test_classes, y_pred_classes)
# plt.figure(figsize=(16,16))
# ax = plt.axes()
# sns.heatmap(cm, annot=True, annot_kws={"size": 30}, fmt='d', cmap="Blues", ax = ax )
# ax.set_title('Confusion Matrix')
# plt.show()

# Get training and test loss histories
training_loss = hist_def.history['loss']
validation_loss = hist_def.history['val_loss']
training_acc = hist_def.history['accuracy']
validation_acc = hist_def.history['val_accuracy']
epoch_count = range(1, len(training_loss) + 1)


score = classificator.evaluate(X_test, y_test, verbose=0)
print(f"Model accuracy on validation data {score[1]*100:.2f}%")

# Display loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, validation_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Display accuracy history
# plt.plot(epoch_count, training_acc, 'r--')
# plt.plot(epoch_count, validation_acc, 'b-')
# plt.legend(['Training Accuracy, Validation Accuracy'])
# plt.xlabel('Epoch', fontsize=14)
# plt.ylabel('Accuracy', fontsize=14)
# plt.show()


print('break')