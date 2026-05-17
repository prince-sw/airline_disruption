import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(filepath='dataset.csv'):
    print("Loading data...")
    df = pd.read_csv(filepath)
    df = df[df['OP_UNIQUE_CARRIER'] == 'DL'].copy()
    df = df[df['CANCELLED'] != 1.0].copy()
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    
    def create_dt(date_series, time_series):
        ts = time_series.fillna(0).astype(int)
        ts = ts.replace(2400, 0)
        hours = ts // 100
        minutes = ts % 100
        return date_series + pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m')
    
    df['scheduled_dep_dt'] = create_dt(df['FL_DATE'], df['CRS_DEP_TIME'])
    df['scheduled_arr_dt'] = create_dt(df['FL_DATE'], df['CRS_ARR_TIME'])
    
    is_overnight = df['scheduled_arr_dt'] < df['scheduled_dep_dt']
    df.loc[is_overnight, 'scheduled_arr_dt'] += pd.Timedelta(days=1)
    
    df = df.sort_values(by='scheduled_dep_dt')
    
    df['prev_scheduled_arr_dt'] = df.groupby('TAIL_NUM')['scheduled_arr_dt'].shift(1)
    df['scheduled_buffer_mins'] = (df['scheduled_dep_dt'] - df['prev_scheduled_arr_dt']).dt.total_seconds() / 60.0
    df['scheduled_buffer_mins'] = df['scheduled_buffer_mins'].fillna(999)
    
    df['prev_leg_arrival_delay_mins'] = df.groupby('TAIL_NUM')['ARR_DELAY'].shift(1)
    df['prev_leg_arrival_delay_mins'] = df['prev_leg_arrival_delay_mins'].fillna(0)
    
    df = df.dropna(subset=['DEP_DELAY'])
    
    df = df.set_index('scheduled_dep_dt')
    df = df.sort_index()
    df['recent_origin_delay'] = df.groupby('ORIGIN')['DEP_DELAY'].transform(
        lambda x: x.rolling('3h', closed='left').mean()
    )
    df = df.reset_index()
    df['recent_origin_delay'] = df['recent_origin_delay'].fillna(0)
    
    df['day'] = df['FL_DATE'].dt.day
    df['dep_hour'] = df['scheduled_dep_dt'].dt.hour
    
    # ---------------------------------------------------------
    # MULTI-CLASS CATEGORIZATION
    # ---------------------------------------------------------
    def categorize_delay(delay):
        if delay <= 15:
            return 0  # On Time / Minor
        elif delay <= 45:
            return 1  # Moderate
        elif delay <= 120:
            return 2  # Severe
        else:
            return 3  # Extreme
            
    df['delay_class'] = df['DEP_DELAY'].apply(categorize_delay)
    
    df = df.sort_values(by='scheduled_dep_dt').reset_index(drop=True)
    df['node_id'] = df.index
    
    return df

def build_graph(df):
    print("Building Graph Edges...")
    edges_source = []
    edges_target = []
    
    df_tail = df.sort_values(['TAIL_NUM', 'scheduled_dep_dt'])
    df_tail['next_node_id'] = df_tail.groupby('TAIL_NUM')['node_id'].shift(-1)
    valid_seq = df_tail.dropna(subset=['next_node_id'])
    edges_source.extend(valid_seq['node_id'].astype(int).tolist())
    edges_target.extend(valid_seq['next_node_id'].astype(int).tolist())
    
    df_sorted = df.sort_values(['ORIGIN', 'scheduled_dep_dt'])
    node_ids = df_sorted['node_id'].values
    origins = df_sorted['ORIGIN'].values
    dep_times = df_sorted['scheduled_dep_dt'].values.astype('datetime64[ns]').astype(np.int64)
    
    fifteen_mins_ns = 15 * 60 * 1000000000
    
    n = len(node_ids)
    for i in range(n):
        orig_i = origins[i]
        time_i = dep_times[i]
        j = i + 1
        while j < n and origins[j] == orig_i and (dep_times[j] - time_i) <= fifteen_mins_ns:
            edges_source.append(node_ids[i])
            edges_target.append(node_ids[j])
            j += 1
            
    edge_index = torch.tensor([edges_source, edges_target], dtype=torch.long)
    
    print("Preparing Node Features...")
    features = ['prev_leg_arrival_delay_mins', 'scheduled_buffer_mins', 'DISTANCE', 'recent_origin_delay', 'dep_hour', 'day']
    X_df = df[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    x = torch.tensor(X_scaled, dtype=torch.float)
    
    # Target is now a long integer representing the class
    y = torch.tensor(df['delay_class'].values, dtype=torch.long)
    
    train_mask = torch.tensor((df['day'] <= 24).values, dtype=torch.bool)
    test_mask = torch.tensor((df['day'] > 24).values, dtype=torch.bool)
    
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    return data, df

class FlightDelayGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FlightDelayGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Raw logits out, CrossEntropyLoss handles softmax internally
        x = self.out(x)
        return x

def train_gnn(data, num_classes=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Out channels now equals number of classes
    model = FlightDelayGNN(in_channels=data.num_features, hidden_channels=32, out_channels=num_classes).to(device)
    
    # We add class weights because 80% of flights are Class 0 (On time). 
    # If we don't, the model will just guess Class 0 for everything!
    class_counts = torch.bincount(data.y[data.train_mask])
    total_train = len(data.y[data.train_mask])
    # Weight = total / (num_classes * class_count)
    class_weights = total_train / (num_classes * class_counts.float())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"Training GNN Multi-Class Classifier on {device}...")
    model.train()
    for epoch in range(1, 251):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch:03d}, Training CrossEntropy Loss: {loss.item():.4f}')
            
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        # Take the class with the highest probability
        pred_classes = out.argmax(dim=1)
        
    return pred_classes.cpu().numpy(), model

def main():
    df = load_and_preprocess_data('dataset.csv')
    print(f"Processed dataset shape: {df.shape}")
    
    data, df = build_graph(df)
    
    pred_classes, model = train_gnn(data, num_classes=4)
    df['pred_delay_class'] = pred_classes
    
    test_df = df[df['day'] > 24]
    y_test = test_df['delay_class'].values
    y_pred = test_df['pred_delay_class'].values
    
    print("\nEvaluation Metrics (GNN - Test Set):")
    class_names = ['On Time (<=15m)', 'Moderate (16-45m)', 'Severe (46-120m)', 'Extreme (>120m)']
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("-" * 50)
    
    print("Generating Confusion Matrix Plot...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['On Time', 'Moderate', 'Severe', 'Extreme'], 
                yticklabels=['On Time', 'Moderate', 'Severe', 'Extreme'])
    
    plt.xlabel('Predicted Delay Severity')
    plt.ylabel('Actual Delay Severity')
    plt.title('GNN Multi-Class Delay Prediction Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_gnn.png')
    print("Saved 'confusion_matrix_gnn.png'")

if __name__ == "__main__":
    main()
