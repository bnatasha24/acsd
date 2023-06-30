import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_stata("/Users/natashabanga/Documents/pandas-intro/STS-ACSD.dta")

# start with predicting mortality using our numerical variables - mortality is a DUMMY variable, looks numeric but possesses
# categorical data
ind_vars = []
num_dep_vars = []
df_drop = df.dropna(0)
for column in df_drop: 
    if column[0:2] == 'p_':
        ind_vars.append(column)
    if column[0:2] == 'o_':
        num_dep_vars.append(column)

X = df_drop[ind_vars]
X = pd.get_dummies(data=X, drop_first=True)
Y = df_drop['o_mortality']

import sklearn
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
# print(pd.DataFrame(model.coef_, X.columns, columns = ['Coeff']))

# to make predictions:
predictions = model.predict(x_test)

# see association between ML model's predictions and the actual values in terms of correlation
# plt.scatter(y_test, predictions)
# plt.hist(y_test - predictions)

# after assessing it visually, we need to test the performance of our linear regression model
from sklearn import metrics
print(metrics.mean_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# getting null data
# df_null_col = df.loc[:, df.isna().any()]
# print(df_null_col.keys())




# distplots for categorical ind. var and numeric dep. var
#plot = sns.FacetGrid(df, hue="p_gender")
#plot.map(sns.distplot, "o_stroke").add_legend()
# 
#plot = sns.FacetGrid(df, hue="p_gender")
#plot.map(sns.distplot, "o_rf").add_legend()
# 
#plot = sns.FacetGrid(df, hue="p_gender")
#plot.map(sns.distplot, "o_vent").add_legend()
# 
#plot = sns.FacetGrid(df, hue="p_gender")
#plot.map(sns.distplot, "o_dswi").add_legend()
# 
#plt.show()

# create new dataframe with only deaths
#dead = df.loc[df['o_mortality'] == 1.0]
#print(df.info())
#sns.countplot(data=dead, x='p_vdinsuft').set(title='Association between severity of tricuspid inefficiency and patient mortality')

# create new dataframe with column consisting of categorical independent variables, another column with all potential categories
# get list of columns with categorical data, find all unique outcomes 
# cat_dict = {}
# cat_df = df.select_dtypes(include='category')

# for col in cat_df:
#     if col[0:2] == 'p_':
#         cat_dict[col] = {}
# for var in cat_dict:
#     for ch in df[var].unique():
#         cat_dict[var][ch] = 0


# # loop through keys and then categories, filter data by category and then again by mortality
# for key in cat_dict:
#     var = key
#     categories = []
#     for category in cat_dict[var]:
#         total = df.loc[df[var] == category]
#         total_count = len(total.index)
#         died = total.loc[total['o_mortality'] == 1.0]
#         died_count = len(died.index)
#         categories.append(category)
#         if total_count != 0:
#             cat_dict[var][category] += died_count / total_count
#         else:
#             cat_dict[var][category] = 0
#     x_vals = list(cat_dict[var].keys())
#     y_vals = list(cat_dict[var].values())
#     sns.barplot(x=x_vals, y=y_vals).set(title=f'{var[2:]} and Mortality')
#     plt.show()

# this code functions like the above code but it has been modified to put the quantitative independent variables against mortality, creating scatter plots

# create new dataframe with column consisting of numeric independent variables, another column with all potential numbers
# get list of columns with numeric data, find all unique outcomes 
# cat_dict = {}
# cat_df = df.select_dtypes(exclude='category')

# for col in cat_df:
#     if col[0:2] == 'p_':
#         cat_dict[col] = {}
# for var in cat_dict:
#     for ch in df[var].unique():
#         cat_dict[var][ch] = 0


# # loop through keys and then values, filter data by values and then again by mortality
# for key in cat_dict:
#     var = key
#     values = []
#     for value in cat_dict[var]:
#         total = df.loc[df[var] == value]
#         total_count = len(total.index)
#         died = total.loc[total['o_mortality'] == 1.0]
#         died_count = len(died.index)
#         values.append(value)
#         if total_count != 0:
#             cat_dict[var][value] += died_count / total_count
#         else:
#             cat_dict[var][value] = 0
#     x_vals = list(cat_dict[var].keys())
#     y_vals = list(cat_dict[var].values())
#     plt.scatter(x=x_vals, y=y_vals, color='red')
#     plt.xlabel(var[2:])
#     plt.ylabel('mortality rate')
#     plt.title(f'Association between {var[2:]} and mortality rate')
#     plt.show()

# tricuspid and mitral insufficiency vs. mortality and vs. other outcomes
# outcomes = []
# for col in df:
#     if col[0:2] == 'o_':
#         outcomes.append(col)
# # loop through keys and then values, filter data by values and then again by outcome with that value
# for outcome in outcomes:
#     x_vals = ['Severe Both', 'Moderate Both', 'Neither']
#     y_vals = []
#     # outcome rate with both (severe)
#     total_severe = df.loc[df['p_vdinsuft'] == 'Severe'].loc[df['p_vdinsufm'] == 'Severe']
#     outcome_severe = total_severe.loc[total_severe[outcome] == 1.0]
#     total_severe_count = len(total_severe.index)
#     outcome_severe_count = len(outcome_severe.index)
#     if total_severe_count != 0:
#         y_vals.append(outcome_severe_count / total_severe_count)
#     else:
#         y_vals.append(0)

#     # outcome with both (moderate)
#     total_moderate = df.loc[df['p_vdinsuft'] == 'Moderate'].loc[df['p_vdinsufm'] == 'Moderate']
#     outcome_moderate = total_moderate.loc[total_moderate[outcome] == 1.0]
#     total_moderate_count = len(total_moderate.index)
#     outcome_moderate_count = len(outcome_moderate.index)
#     if total_moderate_count != 0:
#         y_vals.append(outcome_moderate_count / total_moderate_count)
#     else:
#         y_vals.append(0)

#     # outcome rate with neither
#     total_none = df.loc[df['p_vdinsuft'] == 'None/trace/trivial/mild'].loc[df['p_vdinsufm'] == 'None/trace/trivial/mild']
#     outcome_none = total_none.loc[total_none[outcome] == 1.0]
#     total_none_count = len(total_none.index)
#     outcome_none_count = len(outcome_none.index)
#     if total_none_count != 0:
#         y_vals.append(outcome_none_count / total_none_count)
#     else:
#         y_vals.append(0)
#     # create a barplot of all four
#     sns.barplot(x=x_vals, y=y_vals).set(title=f'Association between Tricuspid and Mitral Insufficiencies and' + outcome[2:])
#     plt.show()

# age vs all outcomes, draw scatter plots
# outcomes = []
# for col in df:
#     if col[0:2] == 'o_':
#         outcomes.append(col)
# # loop through keys and then values, filter data by values and then again by outcome with that value
# for outcome in outcomes:
#     x_vals = list(df['p_age'].unique())
#     y_vals = []
    
#     for age in x_vals:
#         total = df.loc[df['p_age'] == age]
#         total_count = len(total.index)
#         died = total.loc[total[outcome] == 1.0]
#         died_count = len(died.index)
#         if total_count != 0:
#             y_vals.append(died_count / total_count)
#         else:
#             y_vals.append(0)
#     plt.scatter(x=x_vals, y=y_vals, color='red')
#     plt.xlabel('age')
#     plt.ylabel('proportion of patients experiencing ' + outcome[2:])
#     plt.show()


