import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from operator import itemgetter
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, plot_confusion_matrix, \
    precision_score, f1_score, recall_score
from yellowbrick.cluster import KElbowVisualizer
from scipy import stats
from optbinning import ContinuousOptimalBinning
from keras.utils import to_categorical

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

def detect_outliers(df):
    outliers = {}
    for col in df.columns:
        # Calculate the Z-score of each value in the column
        z = np.abs(stats.zscore(df[col]))
        # Identify outliers as values with a Z-score greater than 3
        outliers[col] = np.sum(z > 3)
    return outliers

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
df_final['age_sqrt'] = df_final['age'] ** 2
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


# DEFENDERS
defenders = df_final[df_final['instatPositions'].isin(d)]
defenders.pop('instatPositions')

hcorr_drop(defenders)  # dropping highly correlated variables
dummies_def = pd.get_dummies(defenders['season'], prefix='season')  # SEASONALITY DUMMY
#defenders = pd.concat([defenders, dummies_def], axis=1)
# removing non-relevant variables for the specific role
defenders.drop(['DRB-RX', 'GOL-X', 'OCG', 'PLG-AAX', 'TIR-SX', 'XG'], axis=1, inplace=True)
defenders.sort_values(['playerId'], ascending=[True], inplace=True)
# def_params = list(defenders.columns)
defenders.set_index('playerId', inplace=True)


#====== DEFENDERS CLUSTERING ========
# Find optimal number of clusters
#visualizer_d = KElbowVisualizer(KMeans(init='k-means++'), k=(1, 15)).fit(defenders)  # K=4 best accuracy
# visualizer_d.show()
# Fit KMeans with 4 clusters (optimal number)
kmeans_def = KMeans(n_clusters=3, random_state=0)
clusters_def = (kmeans_def.fit_predict(defenders)) + 1
silhouette_def = metrics.silhouette_score(defenders, kmeans_def.labels_)
defenders['cluster'] = clusters_def

# BAR PLOT
fig = plt.figure()
ax = fig.add_subplot(111)
LABEL_COLOR_MAP = {1: 'r',
                   2: 'k',
                   3: 'b'
                   }

label_color = [LABEL_COLOR_MAP[c] for c in clusters_def]
# scatter = ax.scatter(defenders['marketValue']/1000000, np.sqrt(defenders['age_sqrt']),
#                     c=label_color, s=50)
# ax.set_title('K-Means Clustering')
# ax.set_xlabel('Market Value in millions €')
# ax.set_ylabel('Age')
# plt.colorbar(scatter)
# plt.show()

# summary of clustering after scatterplot --> size, mean of age, mean of vdm, mean of couple of params
cl_size = defenders['cluster'].value_counts()
vdm_mean = defenders.groupby('cluster').marketValue.mean()
age_mean = np.sqrt(defenders.groupby('cluster').age_sqrt.mean())
par_mean = defenders.groupby('cluster').PAR.mean()
drs_mean = defenders.groupby('cluster').DRS.mean()


# ECDF for defenders
# sns.ecdfplot(x=defenders['marketValue']/1000000)
# plt.title('Defenders Market Value ECDF Curve', fontsize=18)
# plt.xlabel('Market Value in millions of €', fontsize=16)
# plt.ylabel('Proportion', fontsize=16)
# plt.legend()
# plt.show()


# defenders binning
bins_def = [0, 2500000, 5000000, 7500000, 10000000,
            15000000, 20000000, 25000000, 30000000, 40000000, 50000000, 80000000]
labels_def = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
defenders_bins = pd.cut(defenders['marketValue'], bins=bins_def, labels=labels_def, precision=0)
def_bins = defenders_bins.value_counts()

def_desc = defenders.describe()
def_outliers = detect_outliers(defenders)
def_describe = defenders.describe()


# defenders_bins = pd.concat(playerid)
# defenders_bins.reset_index(drop=True, inplace=True)
# defenders.reset_index(drop=True, inplace=True)
defenders['vdmbinned'] = defenders_bins

# Fit the model to the data
vdm_defenders = defenders.pop('marketValue')
y_def = defenders.pop('vdmbinned')
y_def = y_def.astype('int')
defenders.pop('cluster')

# balancing through SMOTE algorithm
smote = SMOTE(random_state=42)
defenders_res, y_def_res = smote.fit_resample(defenders, y_def)  # oversampling
defenders = pd.DataFrame(defenders_res, index=defenders.index)
y_def = pd.DataFrame(y_def_res, index=y_def.index)

x_d_train, x_d_test, y_d_train, y_d_test = train_test_split(defenders, y_def, train_size=0.7, random_state=42,
                                                           stratify=y_def)

scaler_def = StandardScaler()
# defenders_scaled = pd.DataFrame(scaler_def.fit_transform(defenders), columns=defenders.columns)
x_d_train = scaler_def.fit_transform(x_d_train)
x_d_test = scaler_def.transform(x_d_test)

### RANDOM FOREST CLASSIFIER --- DEFENDERS ###
# Define the RandomForestClassifier model with class weights

rfc_def = RandomForestClassifier(random_state=42, n_estimators=700, min_samples_leaf=1, min_samples_split=2,
                                 criterion='gini', max_features='auto', max_depth=40, bootstrap=False,
                                 class_weight='balanced')

rfc_def.fit(x_d_train, y_d_train)

# Make predictions on the data
y_pred_def = rfc_def.predict(x_d_test)

# Print accuracy score and confusion matrix
print("Accuracy score:", accuracy_score(y_d_test, y_pred_def))
print("")
print("Recall score:", recall_score(y_d_test, y_pred_def, average='weighted'))
print("")
print("F1 score:", f1_score(y_d_test, y_pred_def, average='weighted'))
print("")
print("Precision score:", precision_score(y_d_test, y_pred_def, average='weighted'))
print(classification_report(y_d_test, y_pred_def))

# conf_mat = confusion_matrix(y_d_test, y_pred_def)
# fig1, ax1 = plt.subplots(figsize=(8, 6))
# plot_confusion_matrix(rfc_def, x_d_test, y_d_test, ax=ax1, cmap=mpl.cm.Blues)  #mpl.cm.Blues
# ax1.set_title("Confusion Matrix")
# plt.show()

y_test_l = list(y_d_test)
y_pred_l = list(y_pred_def)

results = pd.DataFrame({'Actual': y_test_l, 'Predicted': y_pred_l}, index=y_d_test.index).reset_index()
storage = player_comps[['shortName_en', 'playerId']]
results_d = results.rename(columns={'index': 'playerId'})
results_d = results_d.merge(storage, on='playerId')
results.to_excel('def_predRF.xlsx')


def plot_feature_importance(importance,names,model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    data={'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    plt.figure(figsize=(10,8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

# plot_feature_importance(rfc_def.feature_importances_, defenders.columns, 'RANDOM FOREST')



### LOGISTIC REGRESSION - MULTICLASS

# log_reg = LogisticRegression(multi_class='auto', solver='lbfgs', random_state=42, penalty='l2', C=1.0,
#                             max_iter=1000, tol=0.0001)
# lrc = LogisticRegression()
# parameters = {
#    'penalty': ['l1', 'l2', 'elasticnet'],
#    'C': np.logspace(-3, 3, 7),
#    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
#    'max_iter': [1000, 1200, 1500]}

# clf = GridSearchCV(lrc, param_grid=parameters, scoring='accuracy', cv=3)
# clf.fit(x_d_train, y_d_train)
# print(clf.best_params_)
# print(clf.best_score)

# fit(x_d_train, y_d_train)
# log_pred_def = log_reg.predict(x_d_test)
# print("Accuracy score:", accuracy_score(y_d_test, log_pred_def))
# print("")
# print("Recall score:", recall_score(y_d_test, log_pred_def, average='weighted'))
# print("")
# print("F1 score:", f1_score(y_d_test, log_pred_def, average='weighted'))
# print("")
# print("Precision score:", precision_score(y_d_test, log_pred_def, average='weighted'))
# print(classification_report(y_d_test, log_pred_def))


### COMPLEMENT NAIVE BAYES

# complement_NB = ComplementNB(alpha=0.5)
# complement_NB.fit(x_d_train, y_d_train)
# Evaluating the classifier
# pred_cnb_def = complement_NB.predict(x_d_test)
# prediction_train = complement_NB.predict(x_d_train)
# print(f"Training Set Accuracy : {accuracy_score(y_d_train, prediction_train) * 100} %\n")
# print(f"Test Set Accuracy : {accuracy_score(y_d_test, pred_cnb_def) * 100} % \n\n")
# print(f"Complement Naive Bayes Report : \n\n {classification_report(y_d_test, pred_cnb_def)}")


print('break')