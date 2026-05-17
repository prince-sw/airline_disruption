import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
    
    # Calculate buffer
    df['prev_scheduled_arr_dt'] = df.groupby('TAIL_NUM')['scheduled_arr_dt'].shift(1)
    df['scheduled_buffer_mins'] = (df['scheduled_dep_dt'] - df['prev_scheduled_arr_dt']).dt.total_seconds() / 60.0
    df['scheduled_buffer_mins'] = df['scheduled_buffer_mins'].fillna(999)
    
    # Restore the inbound plane delay
    df['prev_leg_arrival_delay_mins'] = df.groupby('TAIL_NUM')['ARR_DELAY'].shift(1)
    df['prev_leg_arrival_delay_mins'] = df['prev_leg_arrival_delay_mins'].fillna(0)
    
    df = df.dropna(subset=['DEP_DELAY'])
    
    # Recent origin delay
    df = df.set_index('scheduled_dep_dt')
    df = df.sort_index()
    df['recent_origin_delay'] = df.groupby('ORIGIN')['DEP_DELAY'].transform(
        lambda x: x.rolling('3h', closed='left').mean()
    )
    df = df.reset_index()
    df['recent_origin_delay'] = df['recent_origin_delay'].fillna(0)
    
    df['day'] = df['FL_DATE'].dt.day
    df['dep_hour'] = df['scheduled_dep_dt'].dt.hour
    
    # Create an explicit node index
    df = df.sort_values(by='scheduled_dep_dt').reset_index(drop=True)
    df['node_id'] = df.index
    
    return df

def build_graph(df):
    print("Building Graph Edges...")
    edges_source = []
    edges_target = []
    
    # 1. Tail Number sequential edges (temporal path of an aircraft)
    print(" - Adding sequential aircraft edges...")
    df_tail = df.sort_values(['TAIL_NUM', 'scheduled_dep_dt'])
    df_tail['next_node_id'] = df_tail.groupby('TAIL_NUM')['node_id'].shift(-1)
    
    valid_seq = df_tail.dropna(subset=['next_node_id'])
    edges_source.extend(valid_seq['node_id'].astype(int).tolist())
    edges_target.extend(valid_seq['next_node_id'].astype(int).tolist())
    
    # 2. Airport Spatial-Temporal Edges (Same ORIGIN, departing within 15 minutes)
    print(" - Adding airport spatial-temporal edges (within 15 mins)...")
    df_sorted = df.sort_values(['ORIGIN', 'scheduled_dep_dt'])
    node_ids = df_sorted['node_id'].values
    origins = df_sorted['ORIGIN'].values
    dep_times = df_sorted['scheduled_dep_dt'].values.astype('datetime64[ns]').astype(np.int64)
    
    fifteen_mins_ns = 15 * 60 * 1000000000
    
    edges_src_spatial = []
    edges_dst_spatial = []
    
    n = len(node_ids)
    for i in range(n):
        orig_i = origins[i]
        time_i = dep_times[i]
        j = i + 1
        while j < n and origins[j] == orig_i and (dep_times[j] - time_i) <= fifteen_mins_ns:
            edges_src_spatial.append(node_ids[i])
            edges_dst_spatial.append(node_ids[j])
            j += 1
            
    edges_source.extend(edges_src_spatial)
    edges_target.extend(edges_dst_spatial)
    
    # STRICTLY DIRECTED EDGES
    src = edges_source
    dst = edges_target
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    print(f"Total directed edges created: {edge_index.size(1)}")
    
    print("Preparing Node Features...")
    features = ['prev_leg_arrival_delay_mins', 'scheduled_buffer_mins', 'DISTANCE', 'recent_origin_delay', 'dep_hour', 'day']
    X_df = df[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    x = torch.tensor(X_scaled, dtype=torch.float)
    
    y = torch.tensor(df['DEP_DELAY'].values, dtype=torch.float).view(-1, 1)
    
    train_mask = torch.tensor((df['day'] <= 24).values, dtype=torch.bool)
    test_mask = torch.tensor((df['day'] > 24).values, dtype=torch.bool)
    
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    return data, df

class FlightDelayGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FlightDelayGNN, self).__init__()
        # Since Graph is directed, we might want to ensure messages flow properly
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.out(x)
        return x

def train_gnn(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    model = FlightDelayGNN(in_channels=data.num_features, hidden_channels=32, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    
    print(f"Training GNN on {device}...")
    model.train()
    for epoch in range(1, 201):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Training MSE Loss: {loss.item():.4f}')
            
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        
    return pred.cpu().numpy(), model

def main():
    df = load_and_preprocess_data('dataset.csv')
    print(f"Processed dataset shape: {df.shape}")
    
    data, df = build_graph(df)
    
    pred_all, model = train_gnn(data)
    df['pred_delay'] = np.maximum(0, pred_all.flatten())
    
    test_df = df[df['day'] > 24]
    y_test = test_df['DEP_DELAY'].values
    y_pred = test_df['pred_delay'].values
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nEvaluation Metrics (GNN - Test Set):")
    print(f"RMSE: {rmse:.2f} mins")
    print(f"MAE:  {mae:.2f} mins")
    print("-" * 50)
    
    print("Generating Plot...")
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='#2ca02c', edgecolor=None)
    
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
    
    plt.xlabel('Actual Departure Delay (mins)')
    plt.ylabel('Predicted Departure Delay (mins)')
    plt.title('Actual vs Predicted Departure Delays (GNN Directed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted_gnn_v2.png')
    print("Saved 'actual_vs_predicted_gnn_v2.png'")

if __name__ == "__main__":
    main()
