import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.feature_selection import SelectKBest, f_regression

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_or_initialize_model(model_class, input_dim, device, model_path='enhanced_rul_model.pth'):
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        model = model_class(input_dim)
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            print(f"Successfully loaded model with input_dim={input_dim}")
        except RuntimeError as e:
            print(f"Error loading model: {e}")
            print(f"Initializing new model due to input dimension mismatch (expected {input_dim} features).")
            model = model_class(input_dim).to(device)
    else:
        print(f"No saved model found. Initializing new model with input_dim={input_dim}.")
        model = model_class(input_dim).to(device)
    return model

col_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]

train_data = pd.read_csv('/Users/harshbhatia/Desktop/SEM 6 MPR/CMaps/train_FD001.txt', sep='\s+', header=None, names=col_names)

train_data['RUL'] = train_data.groupby('engine_id')['cycle'].transform("max") - train_data['cycle']

train_data['RUL'] = train_data['RUL'].clip(upper=125)

# Load CMAPSS test data
test_data = pd.read_csv('/Users/harshbhatia/Desktop/SEM 6 MPR/CMaps/test_FD001.txt', sep='\s+', header=None, names=col_names)
rul_test = pd.read_csv('/Users/harshbhatia/Desktop/SEM 6 MPR/CMaps/RUL_FD001.txt', header=None, names=['RUL'])

max_cycles = test_data.groupby('engine_id')['cycle'].max().reset_index().rename(columns={'cycle': 'cycle_max'})
max_cycles = max_cycles.merge(rul_test, left_index=True, right_index=True, how='left')
test_data = test_data.merge(max_cycles[['engine_id', 'cycle_max', 'RUL']], on='engine_id')
test_data['RUL'] = test_data['cycle_max'] - test_data['cycle'] + test_data['RUL']
# IMPROVED: Apply same RUL clipping to test data for consistency
test_data['RUL'] = test_data['RUL'].clip(upper=125)
test_data = test_data.drop(columns=['cycle_max'])

def add_engineered_features(df):
    key_sensors = [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17]
    for sensor in key_sensors:
        col_name = f'sensor{sensor}'
        if col_name in df.columns:
            # Create squared and cubed terms for nonlinear relationships
            df[f'{col_name}_squared'] = df[col_name] ** 2
            
    # Add window-based statistics (moving averages)
    window_sizes = [5, 10]
    for engine in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine]
        for sensor in key_sensors:
            col_name = f'sensor{sensor}'
            if col_name in df.columns:
                for window in window_sizes:
                    # Calculate rolling statistics
                    rolling_mean = engine_data[col_name].rolling(window=window, min_periods=1).mean()
                    rolling_std = engine_data[col_name].rolling(window=window, min_periods=1).std().fillna(0)
                    
                    idx = engine_data.index
                    df.loc[idx, f'{col_name}_mean_{window}'] = rolling_mean.values
                    df.loc[idx, f'{col_name}_std_{window}'] = rolling_std.values
    
    if 'setting1' in df.columns and 'setting2' in df.columns and 'setting3' in df.columns:
        df['op_severity'] = df['setting1'] * 0.5 + df['setting2'] * 0.3 + df['setting3'] * 0.2
    
    return df

# Apply feature engineering
train_data = add_engineered_features(train_data)
test_data = add_engineered_features(test_data)

X_columns = [col for col in train_data.columns 
             if col not in ['engine_id', 'cycle', 'RUL']]

X = train_data[X_columns]
y = train_data['RUL']

# Apply feature selection using correlation
def select_top_features(X, y, k=40):
    # Use SelectKBest to find top features based on correlation with target
    selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    X_new = selector.fit_transform(X, y)
    
    selected_features = [X.columns[i] for i in selector.get_support(indices=True)]
    print(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
    
    return X[selected_features], selected_features

# Select top features to prevent overfitting
X, selected_features = select_top_features(X, y, k=40)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# IMPROVED: Use MinMaxScaler for RUL which is bounded
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

scaler_y = MinMaxScaler(feature_range=(0, 1))  # Better for bounded RUL
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

print("Summary statistics of RUL (training set):")
print(pd.Series(scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()).describe())

# Convert data to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class EnhancedRULModel(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedRULModel, self).__init__()
        # First block
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        # Third block with residual connection
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        
        # Fourth block
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.act4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)
        
        # Output layer
        self.output = nn.Linear(64, 1)
    
    def forward(self, x):
        # First block
        x1 = self.dropout1(self.act1(self.bn1(self.fc1(x))))
        
        # Second block with residual connection
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = self.act2(x2)
        x2 = self.dropout2(x2)
        x2 = x2 + x1  # Residual connection
        
        # Third block
        x3 = self.dropout3(self.act3(self.bn3(self.fc3(x2))))
        
        # Fourth block
        x4 = self.dropout4(self.act4(self.bn4(self.fc4(x3))))
        
        # Output layer
        return self.output(x4)

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))

model = load_or_initialize_model(EnhancedRULModel, X_train.shape[1], device, model_path='enhanced_rul_model.pth')

criterion = RMSELoss()

if not os.path.exists('enhanced_rul_model.pth') or model.training:
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    epochs = 150  # More epochs with early stopping
    best_val_loss = float('inf')
    patience = 25  # Increased patience
    patience_counter = 0
    
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            # IMPROVED: Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
       
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'enhanced_rul_model.pth')
            print(f"Model improved and saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(torch.load('enhanced_rul_model.pth', map_location=device))
                break

    # Plot training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# IMPROVED: Enhanced evaluation with additional metrics
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_np = y_pred.cpu().numpy().flatten()
    y_test_np = y_test.cpu().numpy().flatten()
    
    
    y_pred_original = scaler_y.inverse_transform(y_pred_np.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test_np.reshape(-1, 1)).flatten()
    
   
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    
    # Calculate RMSE for RUL < 50 (more critical prediction range)
    critical_idx = y_test_original < 50
    if np.sum(critical_idx) > 0:
        critical_rmse = np.sqrt(mean_squared_error(
            y_test_original[critical_idx], 
            y_pred_original[critical_idx]
        ))
    else:
        critical_rmse = float('nan')
    
    # Print comprehensive evaluation metrics
    print("\n=== Model Evaluation ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Critical RMSE (RUL < 50): {critical_rmse:.4f}")
    
    print("\nSample Predictions (Actual vs Predicted RUL):")
    for i in range(10):
        print(f"Actual RUL: {y_test_original[i]:.2f}, Predicted RUL: {y_pred_original[i]:.2f}")

# IMPROVED: More comprehensive visualization
try:
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    # Actual vs Predicted scatter plot
    axs[0, 0].scatter(y_test_original, y_pred_original, alpha=0.5)
    axs[0, 0].plot([0, max(y_test_original)], [0, max(y_test_original)], 'r--')
    axs[0, 0].set_xlabel("Actual RUL")
    axs[0, 0].set_ylabel("Predicted RUL")
    axs[0, 0].set_title("Actual vs Predicted RUL")
    axs[0, 0].grid(True)
    
    # Residual Plot
    residuals = y_test_original - y_pred_original
    axs[0, 1].scatter(y_test_original, residuals, alpha=0.5)
    axs[0, 1].axhline(0, color='red', linestyle='dashed')
    axs[0, 1].set_xlabel("Actual RUL")
    axs[0, 1].set_ylabel("Residual (Actual - Predicted)")
    axs[0, 1].set_title("Residual Plot")
    axs[0, 1].grid(True)
    
    # RUL Distribution Comparison
    sns.kdeplot(y_test_original, label="Actual RUL", ax=axs[1, 0], fill=True, alpha=0.5)
    sns.kdeplot(y_pred_original, label="Predicted RUL", ax=axs[1, 0], fill=True, alpha=0.5)
    axs[1, 0].set_title("RUL Distribution Comparison")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
   
    sns.histplot(residuals, kde=True, ax=axs[1, 1])
    axs[1, 1].axvline(0, color='red', linestyle='dashed')
    axs[1, 1].set_xlabel("Prediction Error")
    axs[1, 1].set_title("Error Distribution")
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
  
    plt.figure(figsize=(16, 6))
    sample_size = min(100, len(y_test_original))
    plt.plot(range(sample_size), y_test_original[:sample_size], 'b-', label="Actual RUL")
    plt.plot(range(sample_size), y_pred_original[:sample_size], 'r--', label="Predicted RUL")
    plt.fill_between(range(sample_size), 
                   y_test_original[:sample_size], 
                   y_pred_original[:sample_size], 
                   alpha=0.2)
    plt.xlabel("Sample Index")
    plt.ylabel("RUL")
    plt.title("Actual vs Predicted RUL Time Series")
    plt.legend()
    plt.grid(True)
    plt.show()
    
except Exception as e:
    print(f"Error in plotting: {e}")

# IMPROVED: Save the processed test data with predictions for further analysis
test_results_df = pd.DataFrame({
    'Actual_RUL': y_test_original,
    'Predicted_RUL': y_pred_original,
    'Error': y_test_original - y_pred_original
})
test_results_df.to_csv('rul_predictions.csv', index=False)
print("Saved predictions to 'rul_predictions.csv'")