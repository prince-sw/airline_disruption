import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(filepath='dataset.csv'):
    print("Loading data...")
    # Read the data
    df = pd.read_csv(filepath)
    
    # 1. Filter by Carrier (e.g., DL for Delta) to avoid memory crash
    print("Filtering for DL flights...")
    df = df[df['OP_UNIQUE_CARRIER'] == 'DL'].copy()
    
    # 2. Handle Cancellations
    # Drop cancelled flights so they don't break the chain sequence math
    print("Dropping cancelled flights...")
    df = df[df['CANCELLED'] != 1.0].copy()
    
    # Convert FL_DATE to datetime
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    
    # 3. Sort Chronologically
    print("Sorting chronologically...")
    df = df.sort_values(by=['FL_DATE', 'CRS_DEP_TIME'])
    
    # 4. Group by Asset (Tail Number) and Date
    print("Grouping by Tail Number and calculating features...")
    # Calculate prev_leg_arrival_delay_mins using shift
    df['prev_leg_arrival_delay_mins'] = df.groupby(['TAIL_NUM', 'FL_DATE'])['ARR_DELAY'].shift(1)
    
    # Calculate scheduled_buffer_mins
    # To do this safely, we convert CRS_DEP_TIME and prev CRS_ARR_TIME to actual datetime objects
    df['prev_leg_crs_arr_time'] = df.groupby(['TAIL_NUM', 'FL_DATE'])['CRS_ARR_TIME'].shift(1)
    
    def convert_to_dt(date_series, time_series):
        # time_series is like 1345.0. Replace NA with 0 and 2400 with 0 (midnight)
        ts = time_series.fillna(0).astype(int)
        ts = ts.replace(2400, 0)
        hours = ts // 100
        minutes = ts % 100
        return date_series + pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m')
    
    df['crs_dep_dt'] = convert_to_dt(df['FL_DATE'], df['CRS_DEP_TIME'])
    df['prev_crs_arr_dt'] = convert_to_dt(df['FL_DATE'], df['prev_leg_crs_arr_time'])
    
    # Buffer is the difference between scheduled departure and previous scheduled arrival in minutes
    df['scheduled_buffer_mins'] = (df['crs_dep_dt'] - df['prev_crs_arr_dt']).dt.total_seconds() / 60.0
    
    # 5. Handle Edge Cases for the First Flight of the Day
    print("Handling edge cases...")
    df['prev_leg_arrival_delay_mins'] = df['prev_leg_arrival_delay_mins'].fillna(0)
    df['scheduled_buffer_mins'] = df['scheduled_buffer_mins'].fillna(999)
    
    # Drop flights with missing target (DEP_DELAY) just in case
    df = df.dropna(subset=['DEP_DELAY'])
    
    return df

def main():
    df = load_and_preprocess_data('dataset.csv')
    print(f"Processed dataset shape: {df.shape}")
    print(df[['FL_DATE', 'TAIL_NUM', 'prev_leg_arrival_delay_mins', 'scheduled_buffer_mins', 'DEP_DELAY']].head())
    print("-" * 50)
    
    print("Training XGBoost Model...")
    
    # Features for the model
    features = ['prev_leg_arrival_delay_mins', 'scheduled_buffer_mins', 'DISTANCE']
    target = 'DEP_DELAY'
    
    # 6. The Chronological Split
    # "If you have 31 days of January data, train your XGBoost model on Days 1 to 24, and test it exclusively on Days 25 to 31."
    print("Splitting data chronologically...")
    df['day'] = df['FL_DATE'].dt.day
    
    train_df = df[df['day'] <= 24]
    test_df = df[df['day'] > 24]
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Initialize and train the XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Evaluation Metrics:")
    print(f"RMSE: {rmse:.2f} mins")
    print(f"MAE:  {mae:.2f} mins")
    print("-" * 50)
    
    print("Generating Plots...")
    
    # Plot 1: Feature Importance
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('XGBoost Feature Importance (Real Data)')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Saved 'feature_importance.png'")
    
    # Plot 2: Actual vs Predicted Scatter Plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='#1f77b4', edgecolor=None)
    
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
    
    plt.xlabel('Actual Departure Delay (mins)')
    plt.ylabel('Predicted Departure Delay (mins)')
    plt.title('Actual vs Predicted Departure Delays (Real Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    print("Saved 'actual_vs_predicted.png'")

if __name__ == "__main__":
    main()
