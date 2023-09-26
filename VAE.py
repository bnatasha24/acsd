import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Data(Dataset):
  def __init__(self, X_train):
    # need to convert float64 to float32 else 
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.X = torch.from_numpy(X_train.astype(np.float32))
    
    # need to convert float64 to Long else 
    # will get the following error
    # RuntimeError: expected scalar type Long but found Float
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index]
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

numericIdxs = []
numeric = ['p_age', 'p_bsa', 'p_bmi', 'p_creatlst', 'p_dialysis', 'p_hct','p_wbc', 'p_platelets', 'p_medadp5days','p_year', 'p_hdef']

for col in categorical:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numeric:
    df[col] = df[col].fillna(df[col].median())
    q_low = df[col].quantile(0.1)
    q_hi  = df[col].quantile(0.9)
    df.loc[df[col] > q_hi, col] = 1010 # tagging the outliers
    df.loc[df[col] < q_low, col] = 1010
    numericIdxs.append(df.columns.get_loc(col))
 
# Instantiate scaler and fit on features
from sklearn.preprocessing import StandardScaler
# deleted preop shock/salvage - only 1% of patients had it
# Split data into features and label 
#df['target'] = df[target]
p_cols = []
for col in df.columns:
    if col[0] == 'p':
        p_cols.append(col) 

#X = pd.concat([df[col] for col in categorical], axis=1)

X = df[p_cols].copy()

# Split data into train and test
x_train, X_tmp = train_test_split(X, train_size=.8, random_state=25)
# add validation set
X_val, X_test = train_test_split(X_tmp, train_size=0.5, random_state=25)

#scale numeric variables
scaler = StandardScaler()
x_train[['p_bsa', 'p_age', 'p_platelets', 'p_creatlst', 'p_status_Elective', 'p_bmi', 'p_hdef', 'p_hct', 'p_wbc', 'p_medadp5days', 'p_year']] = scaler.fit_transform(x_train[['p_bsa', 'p_age', 'p_platelets', 'p_creatlst', 'p_status_Elective', 'p_bmi', 'p_hdef', 'p_hct', 'p_wbc', 'p_medadp5days', 'p_year']])
X_test[['p_bsa', 'p_age', 'p_platelets', 'p_creatlst', 'p_status_Elective', 'p_bmi', 'p_hdef', 'p_hct', 'p_wbc', 'p_medadp5days', 'p_year']] = scaler.fit_transform(X_test[['p_bsa', 'p_age', 'p_platelets', 'p_creatlst', 'p_status_Elective', 'p_bmi', 'p_hdef', 'p_hct', 'p_wbc', 'p_medadp5days', 'p_year']])

torch.manual_seed(25)

traindata = Data(x_train.to_numpy())
testdata = Data(X_test.to_numpy())
batch_size = 512
trainloader = DataLoader(traindata, batch_size=batch_size, 
                         shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testdata, 
                                          batch_size=batch_size, 
                                          shuffle=False)


class Autoencoder(nn.Module):
    def __init__(self, input_dim=130, h1=96, h2=64,latent_dim=32):
        super(Autoencoder,self).__init__()
        
        #Encoder - numeric
        self.linear1 = nn.Linear(input_dim,h1)
        self.lin_bn1 = nn.BatchNorm1d(num_features=h1)
        self.linear2 = nn.Linear(h1,h2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=h2)
        self.linear3 = nn.Linear(h2,h2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=h2)
        
#         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(h2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

#         # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, h2)
        self.fc_bn4 = nn.BatchNorm1d(h2)
        
#         # Decoder
        self.linear4=nn.Linear(h2,h2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=h2)
        self.linear5=nn.Linear(h2,h1)
        self.lin_bn5 = nn.BatchNorm1d(num_features=h1)
        self.linear6=nn.Linear(h1,input_dim)
        self.lin_bn6 = nn.BatchNorm1d(num_features=input_dim)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        
        return r1, r2
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return torch.sigmoid(decoded), mu, logvar

class customLoss(nn.Module):
    def __init__(self, numeric_weight=1.0, categorical_weight=1.0):
        super(customLoss, self).__init__()
        self.numeric_weight = numeric_weight
        self.categorical_weight = categorical_weight
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.bce_loss = nn.BCELoss(reduction="sum")
    
    def forward(self, numeric_preds, numeric_targets, categorical_preds, categorical_targets, mu, logvar):
        loss_numeric = self.numeric_weight * self.mse_loss(numeric_preds, numeric_targets)
        loss_categorical = self.categorical_weight * self.bce_loss(categorical_preds, categorical_targets)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = loss_numeric + loss_categorical + loss_KLD
        return total_loss

# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

def encoder():
    model = Autoencoder()
    model.apply(weights_init_uniform_rule)

    loss_function = customLoss()
 
    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-3,
                             weight_decay = 1e-3)
    
    epochs = 100
    for epoch in range(epochs):
        train_loss = 0.0
        for (data) in trainloader:

            # Output of Autoencoder
            reconstructed, mu, logvar = model(data)

            # split into categorical and quantiative here
            numeric_targets = data[:,0:10]
            numeric_preds = reconstructed[:,0:10]
            
            #slice out numeric columns
            categorical_targets = data[:,11:]
            categorical_preds = reconstructed[:,11:]
            print(categorical_targets)
            
            loss = loss_function(numeric_preds, numeric_targets, categorical_preds, categorical_targets, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        if epoch % 20 == 0: 
            print(f'Epoch: {epoch+1} / {epochs} \t\t\t Training Loss:{train_loss}')

encoder()