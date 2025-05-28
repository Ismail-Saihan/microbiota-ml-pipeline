import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🧬 GPU-ENABLED MICROBIOTA CLASSIFICATION")
    print("=" * 50)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\n📊 Loading data...")
    df = pd.read_csv('advanced_feature_engineered_data.csv')
    print(f"Data shape: {df.shape}")
    
    # Prepare data
    print("🔄 Preparing data...")
    X = df.drop(['Current status of microbiota'], axis=1)
    y = df['Current status of microbiota']
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Classes: {len(np.unique(y_encoded))}")
    
    # Create simple neural network for GPU testing
    class SimpleNet(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(SimpleNet, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Convert to tensors and move to GPU
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create model and move to GPU
    model = SimpleNet(X_train_scaled.shape[1], len(np.unique(y_encoded))).to(device)
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\n🚀 Starting GPU training...")
    model.train()
    epochs = 50
    batch_size = 64
    
    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    print("\n📊 Evaluating model...")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
        # Move back to CPU for sklearn metrics
        y_test_cpu = y_test_tensor.cpu().numpy()
        predicted_cpu = predicted.cpu().numpy()
        
        accuracy = accuracy_score(y_test_cpu, predicted_cpu)
        print(f"✅ Test Accuracy: {accuracy:.4f}")
        
        # Decode predictions for readable report
        y_test_decoded = target_encoder.inverse_transform(y_test_cpu)
        predicted_decoded = target_encoder.inverse_transform(predicted_cpu)
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test_decoded, predicted_decoded))
    
    print(f"\n🎯 GPU Training completed successfully!")
    print(f"Device used: {device}")

if __name__ == "__main__":
    main()
