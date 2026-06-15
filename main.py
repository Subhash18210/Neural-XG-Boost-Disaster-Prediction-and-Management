import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. DATA SIMULATION & PREPROCESSING BLOCK
def generate_mock_disaster_data():
    np.random.seed(42)
    n_samples = 2000
    
    # Simulating data containing missing fields to mimic raw EM-DAT
    deaths = np.random.exponential(scale=100, size=n_samples)
    deaths[np.random.choice(n_samples, size=200, replace=False)] = np.nan
    
    damage = np.random.exponential(scale=5000, size=n_samples)
    damage[np.random.choice(n_samples, size=500, replace=False)] = np.nan
    
    magnitude = np.random.normal(loc=5.5, scale=1.5, size=n_samples)
    
    # Categorical fields
    regions = np.random.choice(['Asia', 'Americas', 'Europe', 'Africa'], size=n_samples)
    regions[np.random.choice(n_samples, size=100, replace=False)] = np.nan
    
    # Imbalanced Target Classes: 70% Floods, 20% Wildfires, 10% Earthquakes
    disaster_types = np.random.choice(['Flood', 'Wildfire', 'Earthquake'], size=n_samples, p=[0.7, 0.2, 0.1])
    
    df = pd.DataFrame({
        'Total_Deaths': deaths,
        'Economic_Damage': damage,
        'Magnitude': magnitude,
        'Region': regions,
        'Disaster_Type': disaster_types
    })
    return df

# Load and process data
df = generate_mock_disaster_data()

# Median/Mode Imputation
num_cols = ['Total_Deaths', 'Economic_Damage', 'Magnitude']
cat_cols = ['Region']

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encoding Categorical Features
le_region = LabelEncoder()
df['Region'] = le_region.fit_transform(df['Region'])

le_target = LabelEncoder()
df['Disaster_Type'] = le_target.fit_transform(df['Disaster_Type'])

X = df.drop(columns=['Disaster_Type']).values
y = df['Disaster_Type'].values

# Train-Test Split (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Z-Score Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Class Balancing via SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 2. PYTORCH NEURAL NETWORK FOR FEATURE EXTRACTION
class FeatureExtractorNN(nn.Module):
    def __init__(self, input_dim, embedding_dim=16):
        super(FeatureExtractorNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, embedding_dim)  # Output feature bottleneck layer
        )
        
    def forward(self, x):
        return self.network(x)

# Convert arrays to PyTorch Tensors
X_train_t = torch.FloatTensor(X_train_res)
y_train_t = torch.LongTensor(y_train_res)
X_test_t = torch.FloatTensor(X_test)

# Train the Feature Extractor
input_dim = X_train.shape[1]
embedding_dim = 8
nn_model = FeatureExtractorNN(input_dim, embedding_dim)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(nn_model.parameters(), lr=0.005)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

nn_model.train()
for epoch in range(30):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = nn_model(batch_x)
        # Using placeholder loss optimization to adjust structural weights
        loss = criterion(outputs[:, :3], batch_y) 
        loss.backward()
        optimizer.step()

# Extract High-Level Embeddings (H)
nn_model.eval()
with torch.no_grad():
    H_train = nn_model(X_train_t).numpy()
    H_test = nn_model(X_test_t).numpy()

# 3. XGBOOST CLASSIFICATION BLOCK
xgb_classifier = XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    objective='multi:softprob',
    random_state=42
)

xgb_classifier.fit(H_train, y_train_res)
y_pred = xgb_classifier.predict(H_test)

# 4. PIPELINE EVALUATION METRICS
print(f"Pipeline Target Evaluation Metrics Reached:")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))
