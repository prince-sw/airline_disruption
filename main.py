import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(filepath='dataset.csv'):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    print("Filtering for DL flights...")
    df = df[df['OP_UNIQUE_CARRIER'] == 'DL'].copy()
    
    print("Dropping cancelled flights...")
    df = df[df['CANCELLED'] != 1.0].copy()
    
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    
    # 1. & 2. Create True unified datetime columns for chronological sorting & calendar math
    def create_dt(date_series, time_series):
        ts = time_series.fillna(0).astype(int)
        ts = ts.replace(2400, 0)
        hours = ts // 100
        minutes = ts % 100
        return date_series + pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m')
    
    df['scheduled_dep_dt'] = create_dt(df['FL_DATE'], df['CRS_DEP_TIME'])
    df['scheduled_arr_dt'] = create_dt(df['FL_DATE'], df['CRS_ARR_TIME'])
    
    # Fix midnight crossing: If arrival is earlier in the day than departure, it crossed midnight
    is_overnight = df['scheduled_arr_dt'] < df['scheduled_dep_dt']
    df.loc[is_overnight, 'scheduled_arr_dt'] += pd.Timedelta(days=1)
    
    # Global Chronological Sort
    print("Sorting chronologically by unified datetime...")
    df = df.sort_values(by='scheduled_dep_dt')
    
    # Global Tail Shifting (crosses days naturally)
    print("Grouping exclusively by TAIL_NUM and shifting...")
    df['prev_leg_arrival_delay_mins'] = df.groupby('TAIL_NUM')['ARR_DELAY'].shift(1)
    df['prev_scheduled_arr_dt'] = df.groupby('TAIL_NUM')['scheduled_arr_dt'].shift(1)
    
    # Buffer calculation using unified true datetimes
    df['scheduled_buffer_mins'] = (df['scheduled_dep_dt'] - df['prev_scheduled_arr_dt']).dt.total_seconds() / 60.0
    
    print("Handling edge cases...")
    df['prev_leg_arrival_delay_mins'] = df['prev_leg_arrival_delay_mins'].fillna(0)
    df['scheduled_buffer_mins'] = df['scheduled_buffer_mins'].fillna(999)
    
    df = df.dropna(subset=['DEP_DELAY'])
    
    # 3. Real-Time Context: Replace static averages with a rolling window 
    print("Engineering real-time rolling airport delay feature...")
    # Set index for rolling window computation
    df = df.set_index('scheduled_dep_dt')
    
    # Calculate the mean departure delay at the ORIGIN airport over the preceding 3 hours.
    # closed='left' ensures the current flight's delay isn't leaked into its own prediction.
    # We sort the index explicitly to be safe, though it should be already sorted.
    df = df.sort_index()
    df['recent_origin_delay'] = df.groupby('ORIGIN')['DEP_DELAY'].transform(
        lambda x: x.rolling('3h', closed='left').mean()
    )
    
    # Reset index and fill missing values (e.g., first flight at an airport)
    df = df.reset_index()
    df['recent_origin_delay'] = df['recent_origin_delay'].fillna(0)
    
    df['day'] = df['FL_DATE'].dt.day
    
    return df

def main():
    df = load_and_preprocess_data('dataset.csv')
    print(f"Processed dataset shape: {df.shape}")
    
    print("Splitting data chronologically...")
    train_df = df[df['day'] <= 24].copy()
    test_df = df[df['day'] > 24].copy()
    
    print("Training XGBoost Model...")
    # Swapped out static features for the dynamic recent_origin_delay
    features = ['prev_leg_arrival_delay_mins', 'scheduled_buffer_mins', 'DISTANCE', 'recent_origin_delay']
    target = 'DEP_DELAY'
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
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
    plt.title('XGBoost Feature Importance (Real-Time Context)')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance_v3.png')
    print("Saved 'feature_importance_v3.png'")
    
    # Plot 2: Actual vs Predicted Scatter Plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='#1f77b4', edgecolor=None)
    
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
    
    plt.xlabel('Actual Departure Delay (mins)')
    plt.ylabel('Predicted Departure Delay (mins)')
    plt.title('Actual vs Predicted Departure Delays (Real-Time Context)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted_v3.png')
    print("Saved 'actual_vs_predicted_v3.png'")

if __name__ == "__main__":
    main()
