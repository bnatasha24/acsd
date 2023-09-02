import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

class Data(Dataset):
  def __init__(self, X_train, y_train):
    # need to convert float64 to float32 else 
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.X = torch.from_numpy(X_train.astype(np.float32))
    # need to convert float64 to Long else 
    # will get the following error
    # RuntimeError: expected scalar type Long but found Float
    self.y = torch.from_numpy(y_train).type(torch.LongTensor)
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  def __len__(self):
    return self.len

# data preprocessing
df_og = pd.read_stata("/Users/natashabanga/Documents/psych and pandas/STS-ACSD.dta")
df_og = df_og.drop('p_preop_iabp', axis=1)

# make dummy variables
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

# Instantiate scaler and fit on features
from sklearn.preprocessing import StandardScaler
# deleted preop shock/salvage - only 1% of patients had it
# Split data into features and label 
#df['target'] = df[target]
p_cols = []
for col in df.columns:
    if col[0] == 'p':
        p_cols.append(col) 
#X = df[['p_bsa', 'p_age', 'p_platelets', 'p_gender_Female', 'p_creatlst', 'p_status_Elective', 'p_bmi']].copy()
X = df[p_cols].copy()
y = df[['o_mortality', 'o_rf', 'o_stroke', 'o_vent', 'o_dswi', 'o_reop', 'o_majorcomposite', 'o_pplos', 'o_splos']].copy() 
from sklearn.model_selection import train_test_split

# Split data into train and test
x_train, X_tmp, y_train, y_tmp = train_test_split(X,
                                                                y,
                                                            train_size=.8,
                                                        random_state=25) # different splits every time code runs
# add validation set
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, train_size=0.5, random_state=25)

# scale numeric variables
scaler = StandardScaler()
x_train[['p_bsa', 'p_age', 'p_platelets', 'p_creatlst', 'p_status_Elective', 'p_bmi']] = scaler.fit_transform(x_train[['p_bsa', 'p_age', 'p_platelets', 'p_creatlst', 'p_status_Elective', 'p_bmi']])
X_test[['p_bsa', 'p_age', 'p_platelets', 'p_creatlst', 'p_status_Elective', 'p_bmi']] = scaler.fit_transform(X_test[['p_bsa', 'p_age', 'p_platelets', 'p_creatlst', 'p_status_Elective', 'p_bmi']])


traindata = Data(x_train.to_numpy(), y_train.to_numpy())
testdata = Data(X_test.to_numpy(), y_test.to_numpy())

batch_size = 200
n_iters = 9000
num_epochs = int(n_iters / (len(traindata) / batch_size))
trainloader = DataLoader(traindata, batch_size=batch_size, 
                         shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testdata, 
                                          batch_size=batch_size, 
                                          shuffle=False)
print(trainloader)
def NeuralNetwork():
    class MultiOutputClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(MultiOutputClassifier, self).__init__()
            
            # Define the layers of your neural network
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            # Forward pass through the network
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    # instantiate the multioutput classifier
    input_size = 130
    hidden_size = 100
    num_classes = 9
    model = MultiOutputClassifier(input_size, hidden_size, num_classes)

    # instantiate the loss class
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1) 

    # Assuming you have a DataLoader with your dataset
    for epoch in range(num_epochs):
        for inputs, targets in trainloader:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate the loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update the weights
            optimizer.step()

    PATH = '/Users/natashabanga/Documents/eda_code/acsd/testFNN.pth'
    torch.save(model.state_dict(), PATH)

