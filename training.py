import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import category_encoders as cat_encoder
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, truncnorm, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pprint import pprint
import xgboost as xgb
from sklearn.utils import resample




def XGB():
    
  # # XGBoost
  
  from sklearn.metrics import precision_recall_fscore_support as score
  from sklearn.metrics import accuracy_score
  from sklearn.metrics import classification_report
  xgb_model = xgb.XGBClassifier(objective="binary:logistic", reg_lambda=1, reg_alpha=1e-05, max_depth=12, learning_rate=0.001, gamma=0.1, colsample_bytree=0.9, eta=0.1, subsample=0.5, enable_categorical=False, min_child_weight=1, n_estimators=100, scale_pos_weight=75)
  xgb_model.fit(x_train_scaled.to_numpy(), y_train_scaled.to_numpy())

  y_pred = xgb_model.predict(X_test)

  result = confusion_matrix(y_test, y_pred)
  print("Confusion Matrix:")
  print(result)

  result1 = classification_report(y_test, y_pred)
  print("Classification Report:",)
  print (result1)

  # # Get performance metrics
  precision, recall, fscore, support = score(y_test, y_pred)
  # # Print result
  print(f'The recall value for the baseline xgboost model is {recall[1]:.4f}')
  print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
  ROC(xgb_model, 'XGB on test dataset')

def randomSearch():

  xgb_model = xgb.XGBClassifier()

  param_grid = { 
    # Learning rate shrinks the weights to make the boosting process more conservative
      "learning_rate": [0.0001,0.001, 0.01, 0.1, 1] ,
      # Maximum depth of the tree, increasing it increases the model complexity.
      "max_depth": range(3,21,3),
      # Gamma specifies the minimum loss reduction required to make a split.
      "gamma": [i/10.0 for i in range(0,5)],
      # Percentage of columns to be randomly samples for each tree.
      "colsample_bytree": [i/10.0 for i in range(3,10)],
      # reg_alpha provides l1 regularization to the weight, higher values result in more conservative models
      "reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100],
      # reg_lambda provides l2 regularization to the weight, higher values result in more conservative models
      "reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100],
      'min_child_weight': [0, 1, 100, 1000, 10000],
      'subsample': [0, 0.25, 0.5, 0.75, 1],
      'scale_pos_weight': [1, 10, 25, 50, 75, 99, 100, 1000]



    # 'n_estimators': randint(4,200),
    # 'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
    # 'min_samples_split': uniform(0.01, 0.199),
    # #'max_depth': randint(1, 20)
  }
    
  
  xgboost = RandomizedSearchCV(xgb_model, param_grid, n_iter=100, cv=5, random_state=1)
  model = xgboost.fit(X_train_scaled, y_train)
  predictions = model.predict(X_test_scaled)

  pprint(model.best_estimator_.get_params())
  result = accuracy_score(y_test, predictions)
  print("Accuracy:",result)
  




def randomSearchCross(model):
    
  # # randomized search
  from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
  # # Define the search space
  param_grid = { 
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9]
    }
      # Learning rate shrinks the weights to make the boosting process more conservative
      # "learning_rate": [0.0001,0.001, 0.01, 0.1, 1] ,
      # # Maximum depth of the tree, increasing it increases the model complexity.
      # "max_depth": range(3,21,3),
      # # Gamma specifies the minimum loss reduction required to make a split.
      # "gamma": [i/10.0 for i in range(0,5)],
      # # Percentage of columns to be randomly samples for each tree.
      # "colsample_bytree": [i/10.0 for i in range(3,10)],
      # # reg_alpha provides l1 regularization to the weight, higher values result in more conservative models
      # "reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100],
      # # reg_lambda provides l2 regularization to the weight, higher values result in more conservative models
      # "reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100]}
  # # Set up score
  scoring = ['recall']
  # Set up the k-fold cross-validation
  kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

  # # Define random search
  random_search = RandomizedSearchCV(estimator=model, 
                            param_distributions=param_grid, 
                            n_iter=48,
                            scoring=scoring, 
                            refit='recall', 
                            n_jobs=-1, 
                            cv=kfold, 
                            verbose=0)
  # Fit grid search
  random_result = random_search.fit(X_train_scaled, y)
  # Print grid search summary
  print(random_result)

  # Print the best score and the corresponding hyperparameters
  print(f'The best score is {random_result.best_score_:.4f}')
  print('The best score standard deviation is', round(random_result.cv_results_['std_test_recall'][random_result.best_index_], 4))
  print(f'The best hyperparameters are {random_result.best_params_}')

#   {'bootstrap': True,
#  'ccp_alpha': 0.0,
#  'class_weight': None,
#  'criterion': 'gini',
#  'max_depth': None,
#  'max_features': 0.2864742729236049,
#  'max_leaf_nodes': None,
#  'max_samples': None,
#  'min_impurity_decrease': 0.0,
#  'min_samples_leaf': 1,
#  'min_samples_split': 0.15334457419498948,
#  'min_weight_fraction_leaf': 0.0,
#  'n_estimators': 141,
#  'n_jobs': None,
#  'oob_score': False,
#  'random_state': None,
#  'verbose': 0,
#  'warm_start': False}

def gridSearch(model):
  param_grid = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
  }

  # hyperparamater tuning for random forest
  from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

  # using grid search
  grid_search = GridSearchCV(model,
                            param_grid=param_grid)
  grid_search.fit(X_train_scaled, y_train)
  print(grid_search.best_estimator_)

def ROC(model, name):
  # ROC curve
  from sklearn import metrics
  from sklearn.metrics import confusion_matrix
  import matplotlib.pyplot as plt
  from sklearn.metrics import roc_auc_score, roc_curve

  probs_xg = model.predict_proba(X_test)[:, 1] # calculate predictive probability
  y_test_int = y_test.replace({'Good': 1, 'Bad': 0})
  auc_xg = roc_auc_score(y_test_int, probs_xg)
  fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test_int, probs_xg)
  plt.figure(figsize=(12, 7))
  plt.plot(fpr_xg, tpr_xg, label=f'AUC ({name}) = {auc_xg:.2f}')
  plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
  plt.title('ROC Curve', size=20)
  plt.xlabel('False Positive Rate', size=14)
  plt.ylabel('True Positive Rate', size=14)
  plt.legend()
  plt.show()

def randomForest(predictX, predictY, note):
  # random forest classifier - key paramaters are max features and n estimators
  # max features is number of subsets of features to consider when splitting a node
  # # estimators are the number of trees in the forest - more trees is better
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import classification_report
  from sklearn.metrics import accuracy_score
  from sklearn.metrics import confusion_matrix
  

  RFclf = RandomForestClassifier(bootstrap=True,
 ccp_alpha=0.0,
 class_weight=None,
 criterion='gini',
 max_depth=None,
 max_features=0.2864742729236049,
 max_leaf_nodes=None,
 max_samples=None,
 min_impurity_decrease=0.0,
 min_samples_leaf=1,
 min_samples_split=0.15334457419498948,
 min_weight_fraction_leaf=0.0,
 n_estimators=141,
 n_jobs=None,
 oob_score=False,
 random_state=None,
 verbose=0,
 warm_start=False)
  RFclf.fit(X_train_scaled.values, y_train)

  y_pred = RFclf.predict(predictX)

  result = confusion_matrix(predictY, y_pred)
  print("Confusion Matrix:")
  print(result)
  result1 = classification_report(predictY, y_pred)
  print("Classification Report:",)
  print (result1)
  result2 = accuracy_score(predictY, y_pred)
  print("Accuracy:",result2)
  
  ROC(RFclf, f'Random Forest {note}')


def featureImp(model):
  # feature importance analysis
  from sklearn.inspection import permutation_importance
  use = df[p_cols]
  sort = model.feature_importances_.argsort()[-30:]
  plt.barh(use.columns[sort], model.feature_importances_[sort])
  plt.xlabel("Feature Importance")
  plt.title("Top 30 Features for Mortality")
  plt.show()

def basics():
  # logistic regression, svm, and decision tree (classifier)
  from sklearn.linear_model import LogisticRegression
  from sklearn.svm import SVC
  from sklearn.tree import DecisionTreeClassifier

  # Instnatiating the models 
  logistic_regression = LogisticRegression()
  svm = SVC(probability=True)
  tree = DecisionTreeClassifier()

  # Training the models 
  logistic_regression.fit(X_train_scaled, y_train)
  svm.fit(X_train_scaled, y_train)
  tree.fit(X_train_scaled, y_train)

  # Making predictions with each model
  log_reg_preds = logistic_regression.predict(X_test_scaled)
  svm_preds = svm.predict(X_test_scaled)
  tree_preds = tree.predict(X_test_scaled)

  from sklearn.metrics import classification_report

  # Store model predictions in a dictionary
  # this makes it's easier to iterate through each model
  # and print the results. 
  model_preds = {
      "Logistic Regression": log_reg_preds,
      # "Support Vector Machine": svm_preds,
      # "Decision Tree": tree_preds
  }

  for model, preds in model_preds.items():
      result = confusion_matrix(y_test, preds)
      print("Confusion Matrix:")
      print(result)
      print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")
      
  





df_og = pd.read_stata("/Users/natashabanga/Documents/psych and pandas/STS-ACSD.dta")
df_og = df_og.drop('p_preop_iabp', axis=1)


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

# creating an object BinaryEncoder
# this code calls all columns0
# we can specify specific columns as well
# encoder = cat_encoder.BinaryEncoder(cols = ['p_gender', 'p_chrlungd', 'p_medster', 'p_vdinsufa', 'p_vdinsufm',
#        'p_vdinsuft', 'p_arrhythmia', 'p_endocarditis', 'p_cvd', 'p_carsten',
#        'p_alcohol', 'p_drugab', 'p_pneumonia', 'p_mediastrad', 'p_cancer',
#        'p_diabetes', 'p_numdisv', 'p_prevmi', 'p_presentation', 'p_race',
#        'p_status', 'p_acei', 'p_chf', 'p_smoker', 'p_fhcad', 'p_homeo2',
#        'p_osa', 'p_liver', 'p_unresponsive', 'p_syncope', 'p_prevcab',
#        'p_prevav', 'p_prevmv', 'p_prevtc', 'p_prevothervalve', 'p_previcd',
#        'p_pci', 'p_prevothercardiac', 'p_payor', 'p_preop_shock', 'p_vdstenm'])

# df= encoder.fit_transform(df_og)
# 'p_gender', 'p_hypertn', 'p_immsupp', 'p_medster', 'p_medgp', 'p_medinotr', 'p_preop_iabp', 'p_preop_shock', 'p_pvd', 'p_lm', 'p_lad', 'p_rootabscess', 'p_vdstenm', 'p_vdstena', 'p_drugab', 'p_pneumonia', 'p_mediastrad', 'p_cancer', 'p_acei', 'p_fhcad', 'p_homeo2', 'p_osa', 'p_liver', 'p_unresponsive', 'p_syncope', 'p_prevcab', 'p_prevav', 'p_prevmv', 'p_prevtc', 'p_prevothervalve', 'p_prevothercardiac'

# 'p_vdinsufa', 'p_vdinsufm', 'p_vdinsuft', 'p_arrhythmia', 'p_endocarditis', 'p_chrlungd', 'p_cvd', 'p_carsten', 'p_alcohol', 'p_diabetes', 'p_numdisv', 'p_prevmi', 'p_presentation', 'p_race', 'p_status', 'p_chf', 'p_smoker', 'p_numcvsurg', 'p_pci', 'p_payer'

# dummy columns for binary variables
df_updated = pd.get_dummies(data=df_og, columns=['p_gender', 'p_hypertn', 'p_immsupp', 'p_medster', 
                                 'p_medgp', 'p_medinotr', 'p_preop_shock', 
                                 'p_pvd', 'p_lm', 'p_lad', 'p_rootabscess', 'p_vdstenm', 
                                 'p_vdstena', 'p_drugab', 'p_pneumonia', 'p_mediastrad', 
                                 'p_cancer', 'p_acei', 'p_fhcad', 'p_homeo2', 'p_osa', 
                                 'p_liver', 'p_unresponsive', 'p_syncope', 'p_prevcab', 
                                 'p_prevav', 'p_prevmv', 'p_prevtc', 'p_prevothervalve', 
                                 'p_prevothercardiac', 'p_previcd'], drop_first=True)


# dummy columns for categorical variables
df = pd.get_dummies(data=df_updated, columns=['p_vdinsufa', 'p_vdinsufm', 'p_vdinsuft', 'p_arrhythmia', 
                                 'p_endocarditis', 'p_chrlungd', 'p_cvd', 'p_carsten', 'p_alcohol', 
                                 'p_diabetes', 'p_numdisv', 'p_prevmi', 'p_presentation', 'p_race', 
                                 'p_status', 'p_chf', 'p_smoker', 'p_numcvsurg', 'p_pci', 'p_payor'])

# data imputation
from sklearn.impute import SimpleImputer
categorical = [
               'p_gender_Female', 'p_hypertn_1.0', 
               'p_immsupp_1.0', 'p_medster_Yes', 'p_medgp_1.0', 'p_medinotr_1.0', 
               'p_preop_shock_1.0', 'p_pvd_1.0', 'p_lm_1', 'p_lad_1', 'p_rootabscess_1', 'p_vdstenm_1', 
               'p_vdstena_1', 'p_drugab_Yes', 'p_pneumonia_Yes', 'p_mediastrad_Yes', 'p_cancer_Yes', 
               'p_acei_Yes', 'p_fhcad_Yes', 'p_osa_Yes', 'p_liver_Yes', 'p_unresponsive_Yes', 'p_syncope_Yes', 
               'p_prevcab_Yes', 'p_prevav_Yes', 'p_prevmv_Yes', 'p_prevtc_Yes', 'p_prevothervalve_Yes', 
               'p_prevothercardiac_Yes', 'p_previcd_Yes', 'p_vdinsufa_None/trace/mild', 'p_vdinsufa_Moderate', 
               'p_vdinsufa_Severe', 'p_vdinsufm_None/trace/mild', 'p_vdinsufm_Moderate', 'p_vdinsufm_Severe', 
               'p_vdinsuft_None/trace/mild', 'p_vdinsuft_Moderate', 'p_vdinsuft_Severe', 'p_arrhythmia_Recent continuous afib/flutter', 
               'p_arrhythmia_Recent paroxysmal afib/flutter', 'p_arrhythmia_Recent 3rd degree block', 'p_arrhythmia_Recent 2nd degree block or SSS', 
               'p_arrhythmia_Recent VT/VF', 'p_arrhythmia_Remote arrhythmia', 'p_arrhythmia_No arrhythmia', 'p_endocarditis_None', 
               'p_endocarditis_Treated endocarditis', 'p_endocarditis_Active endocarditis', 'p_chrlungd_None', 'p_chrlungd_Mild', 
               'p_chrlungd_Moderate', 'p_chrlungd_Severe', 'p_chrlungd_Present, unknown severity', 'p_cvd_None', 'p_cvd_CVD no TIA/CVA', 
               'p_cvd_CVD + TIA', 'p_cvd_CVD + remote CVA', 'p_cvd_CVD + recent CVA', 'p_carsten_None', 'p_carsten_One side', 'p_carsten_Both sides', 
               'p_alcohol_<= 1 drink/week', 'p_alcohol_2-7 drinks/week', 'p_alcohol_8+ drinks/week', 'p_diabetes_None', 'p_diabetes_DM - no control', 'p_diabetes_DM - diet or other', 'p_diabetes_DM - oral control', 'p_diabetes_DM - insulin', 'p_numdisv_Less than 2', 'p_numdisv_2', 'p_numdisv_3', 'p_prevmi_<= 6 hours', 'p_prevmi_6-24 hours', 'p_prevmi_1-21 days', 'p_prevmi_> 21 days or no MI', 'p_presentation_No ischemia', 'p_presentation_Stable angina', 'p_presentation_Unstable angina', 'p_presentation_NSTEMI', 'p_presentation_STEMI', 'p_race_Asian', 'p_race_Black', 'p_race_Hispanic', 'p_race_Native American', 'p_race_Pacific Islander', 'p_race_Other, including non-Hispanic white', 'p_status_Elective', 'p_status_Urgent', 'p_status_Emergent, no resuscitation', 'p_status_Salvage', 'p_chf_None', 'p_chf_CHF but > 2 weeks', 'p_chf_CHF within 2 weeks, not NYHA 4', 'p_chf_CHF within 2 weeks, NYHA 4', 'p_smoker_No smoking within 30 days', 'p_smoker_Smoker', 'p_smoker_Former smoker', 'p_numcvsurg_0', 'p_numcvsurg_1', 'p_numcvsurg_2', 'p_numcvsurg_3', 'p_numcvsurg_4', 'p_numcvsurg_5', 'p_numcvsurg_6', 'p_pci_No prior PCI', 'p_pci_Prior PCI but not during this care episode', 'p_pci_Prior PCI > 6 hours from surgery', 'p_pci_Prior PCI within 6 hours of surgery', 'p_payor_Age 65+, Medicare + Medicaid, dual eligible', 'p_payor_Age 65+, Medicare', 'p_payor_Age 65+, Commercial or HMO', 'p_payor_Age 65+, Medicaid/Other', 'p_payor_Age <65, Medicaid + Medicare', 'p_payor_Age <65, Medicaid', 'p_payor_Age <65, Medicare', 'p_payor_Age <65, Commercial or HMO', 'p_payor_Age <65, Self/none as only payor']

numeric = ['p_age', 'p_bsa', 'p_bmi', 'p_creatlst', 'p_dialysis', 'p_hct','p_wbc', 'p_platelets', 'p_medadp5days','p_year', 'p_hdef']



for col in categorical:
  df[col] = df[col].fillna(df[col].mode()[0])

for col in numeric:
  df[col] = df[col].fillna(df[col].median())
  q_low = df[col].quantile(0.1)
  q_hi  = df[col].quantile(0.9)
  df.loc[df[col] > q_hi, col] = 1010 # tagging the outliers
  df.loc[df[col] < q_low, col] = 1010

for col in numeric:
  df = df.loc[df[col] != 1010]

# 243,173 patients after filtering outliers


# Instantiate scaler and fit on features
from sklearn.preprocessing import StandardScaler
# deleted preop shock/salvage - only 1% of patients had it
# Split data into features and label 
df['target'] = df['o_mortality']
p_cols = []
for col in df.columns:
    if col[0] == 'p':
        p_cols.append(col) 
X = df[['p_bsa', 'p_age', 'p_platelets', 'p_gender_Female', 'p_creatlst', 'p_status_Elective', 'p_bmi']].copy()
# X = df[p_cols].copy()
y = df["target"].copy() 




from sklearn.model_selection import train_test_split

# Split data into train and test
x_train, X_tmp, y_train, y_tmp = train_test_split(X,
                                                                  y,
                                                             train_size=.8,
                                                           random_state=25) # different splits every time code runs
# add validation set
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, train_size=0.5, random_state=25)


# normalize data, using same parameters as train on test set
scaler = StandardScaler()
x_train[['p_bsa', 'p_age', 'p_platelets', 'p_creatlst', 'p_status_Elective', 'p_bmi']] = scaler.fit_transform(x_train[['p_bsa', 'p_age', 'p_platelets', 'p_creatlst', 'p_status_Elective', 'p_bmi']])
X_test[['p_bsa', 'p_age', 'p_platelets', 'p_creatlst', 'p_status_Elective', 'p_bmi']] = scaler.fit_transform(X_test[['p_bsa', 'p_age', 'p_platelets', 'p_creatlst', 'p_status_Elective', 'p_bmi']])



# resample training set
train_df = pd.concat([x_train, y_train], axis=1)

survived = train_df.loc[train_df['target'] == 0]
dead = train_df.loc[train_df['target'] == 1]

#upsample the minority class
dead_upsampled = resample(dead,random_state=42,n_samples=3000,replace=True)
survived_downsampled = resample(survived,random_state=42,n_samples=3000,replace=True)

# Combine minority class with upsampled minority class
df_mort = pd.concat([dead_upsampled, survived_downsampled])

y_train_scaled = df_mort['target']
x_train_scaled = df_mort.drop(['target'], axis=1)

XGB()





