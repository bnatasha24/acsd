import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import category_encoders as cat_encoder

df = pd.read_stata("/Users/natashabanga/Documents/pandas-intro/STS-ACSD.dta")
df_og = pd.read_stata("/Users/natashabanga/Documents/pandas-intro/STS-ACSD.dta").dropna()

# looking at feature names
# print(df_og.head())
# print(list(df_og.columns))

# find features that need to be converted to dummy variables
# print(df_og.select_dtypes(exclude=np.number).columns)

# percentage per column with missing data
# for col in df.columns:
#    print(col)
#    print(df[col].isna().sum())

# assess incidence of major composite
# print((df_og['o_majorcomposite'] == 1.0).sum())
# print((df_og['o_majorcomposite'] == 0.0).sum())

# # assess years of surgeries
# print(df_og['p_year'].min())
# print(df_og['p_year'].max())

# creating 2x2 table of mortality and mp performance
# print(df_og['mp_predmort'])
# print((df_og['o_mortality'] == 1.0).sum())
# print((df_og['o_mortality'] == 0.0).sum())
# print((df_og['mp_predmort'] == 1.0).sum())
# print((df_og['mp_predmort'] == 0.0).sum())

# creating dummy predictors
cat_columns = df_og.select_dtypes(exclude=np.number).columns

# creating an object BinaryEncoder
# this code calls all columns0
# we can specify specific columns as well
encoder = cat_encoder.BinaryEncoder(cols = ['p_gender', 'p_chrlungd', 'p_medster', 'p_vdinsufa', 'p_vdinsufm',
       'p_vdinsuft', 'p_arrhythmia', 'p_endocarditis', 'p_cvd', 'p_carsten',
       'p_alcohol', 'p_drugab', 'p_pneumonia', 'p_mediastrad', 'p_cancer',
       'p_diabetes', 'p_numdisv', 'p_prevmi', 'p_presentation', 'p_race',
       'p_status', 'p_acei', 'p_chf', 'p_smoker', 'p_fhcad', 'p_homeo2',
       'p_osa', 'p_liver', 'p_unresponsive', 'p_syncope', 'p_prevcab',
       'p_prevav', 'p_prevmv', 'p_prevtc', 'p_prevothervalve', 'p_previcd',
       'p_pci', 'p_prevothercardiac', 'p_payor', 'p_preop_shock', 'p_vdstenm'])
df= encoder.fit_transform(df_og)
df["target"] = df["o_majorcomposite"]
# using estimator/clf
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)


from sklearn.preprocessing import StandardScaler

# Split data into features and label 
# p_cols = []
# for col in df.columns:
#     if col[0] == 'p':
#         p_cols.append(col)
# X = df[p_cols].copy()
X = df[['p_bmi', 'p_creatlst', 'p_platelets', 'p_wbc' , 'p_preop_shock_1', 'p_smoker_1', 'p_vdstenm_1', 'p_hdef']].copy()
y = df["target"].copy() 

# Instantiate scaler and fit on features
scaler = StandardScaler()
scaler.fit(X)
#
# Transform features
X_scaled = scaler.transform(X.values)


# View first instance
# print(X_scaled[0])


from sklearn.model_selection import train_test_split

# Split data into train and test
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled,
                                                                  y,
                                                             train_size=.85,
                                                           random_state=25)

# Check the splits are correct
# print(X_train_scaled)
# print(X_test_scaled)

# # XGBoost
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support as score
xgb_model = xgb.XGBClassifier(objective="binary:logistic")
xgb_model.fit(X_train_scaled, y_train)

y_pred = xgb_model.predict(X_test_scaled)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# result1 = classification_report(y_test, y_pred)
# print("Classification Report:",)
# print (result1)

# # Get performance metrics
# precision, recall, fscore, support = score(y_test, y_pred)
# # Print result
# print(f'The recall value for the baseline xgboost model is {recall[1]:.4f}')

# # randomized search

# # Define the search space
# param_grid = { 
#     # Learning rate shrinks the weights to make the boosting process more conservative
#     "learning_rate": [0.0001,0.001, 0.01, 0.1, 1] ,
#     # Maximum depth of the tree, increasing it increases the model complexity.
#     "max_depth": range(3,21,3),
#     # Gamma specifies the minimum loss reduction required to make a split.
#     "gamma": [i/10.0 for i in range(0,5)],
#     # Percentage of columns to be randomly samples for each tree.
#     "colsample_bytree": [i/10.0 for i in range(3,10)],
#     # reg_alpha provides l1 regularization to the weight, higher values result in more conservative models
#     "reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100],
#     # reg_lambda provides l2 regularization to the weight, higher values result in more conservative models
#     "reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100]}
# # # Set up score
# # scoring = ['recall']
# # # Set up the k-fold cross-validation
# # kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# # # Define random search
# # random_search = RandomizedSearchCV(estimator=xgb_model, 
# #                            param_distributions=param_grid, 
# #                            n_iter=48,
# #                            scoring=scoring, 
# #                            refit='recall', 
#                            n_jobs=-1, 
#                            cv=kfold, 
#                            verbose=0)
# # Fit grid search
# random_result = random_search.fit(X_train_scaled, y_train)
# # Print grid search summary
# print(random_result)

# # Print the best score and the corresponding hyperparameters
# print(f'The best score is {random_result.best_score_:.4f}')
# print('The best score standard deviation is', round(random_result.cv_results_['std_test_recall'][random_result.best_index_], 4))
# print(f'The best hyperparameters are {random_result.best_params_}')

# ROC curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score, roc_curve

# probs_xg = xgb_model.predict_proba(X_test_scaled)[:, 1] # calculate predictive probability
# y_test_int = y_test.replace({'Good': 1, 'Bad': 0})
# auc_xg = roc_auc_score(y_test_int, probs_xg)
# fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test_int, probs_xg)
# plt.figure(figsize=(12, 7))
# plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost) = {auc_xg:.2f}')
# plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
# plt.title('ROC Curve', size=20)
# plt.xlabel('False Positive Rate', size=14)
# plt.ylabel('True Positive Rate', size=14)
# plt.legend()
# plt.show()


print(confusion_matrix(y_test, y_pred))

# from sklearn.metrics import accuracy_score

print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# random forest classifier - key paramaters are max features and n estimators
# max features is number of subsets of features to consider when splitting a node
# # estimators are the number of trees in the forest - more trees is better
from sklearn.ensemble import RandomForestClassifier

RFclf = RandomForestClassifier(max_depth=3, max_leaf_nodes=6, n_estimators=150)
RFclf.fit(X_train_scaled, y_train)
y_pred = RFclf.predict(X_test_scaled)
print(y_pred[0:10])

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
# param_grid = {
#     'n_estimators': [25, 50, 100, 150],
#     'max_features': ['sqrt', 'log2', None],
#     'max_depth': [3, 6, 9],
#     'max_leaf_nodes': [3, 6, 9],
# }

# hyperparamater tuning for random forest
# from sklearn.model_selection import GridSearchCV,\
# RandomizedSearchCV

# # using grid search
# grid_search = GridSearchCV(RandomForestClassifier(),
#                            param_grid=param_grid)
# grid_search.fit(X_train_scaled, y_train)
# print(grid_search.best_estimator_)

# using randomized search
# random_search = RandomizedSearchCV(RandomForestClassifier(),
#                                    param_grid)
# random_search.fit(X_train_scaled, y_train)
# print(random_search.best_estimator_)


# feature importance analysis of random forest
from sklearn.inspection import permutation_importance
sort = RFclf.feature_importances_.argsort()
plt.barh(df.columns[sort], RFclf.feature_importances_[sort])
plt.xlabel("Feature Importance")
# plt.figure().subplots_adjust(left=0.01)
plt.show()


# logistic regression, svm, and decision tree (classifier)
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier

# Instnatiating the models 
# logistic_regression = LogisticRegression()
# svm = SVC()
# tree = DecisionTreeClassifier()

# # Training the models 
# logistic_regression.fit(X_train_scaled, y_train)
# svm.fit(X_train_scaled, y_train)
# tree.fit(X_train_scaled, y_train)

# # Making predictions with each model
# log_reg_preds = logistic_regression.predict(X_test_scaled)
# svm_preds = svm.predict(X_test_scaled)
# tree_preds = tree.predict(X_test_scaled)


# from sklearn.metrics import classification_report

# # Store model predictions in a dictionary
# # this makes it's easier to iterate through each model
# # and print the results. 
# model_preds = {
#     "Logistic Regression": log_reg_preds,
#     "Support Vector Machine": svm_preds,
#     "Decision Tree": tree_preds
# }

# for model, preds in model_preds.items():
#     print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")

