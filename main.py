import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def generate_data(n_samples=10000, seed=42):
    """Generates synthetic dataset for flight transitions."""
    np.random.seed(seed)
    
    tail_numbers = [f'N{i:03d}' for i in range(100, 300)]
    
    # prev_delay generated using exponential distribution to mimic real world 
    # where most delays are small, but there's a long tail of severe delays.
    data = {
        'tail_number': np.random.choice(tail_numbers, n_samples),
        'prev_leg_arrival_delay_mins': np.random.exponential(scale=25, size=n_samples), 
        'scheduled_buffer_mins': np.random.uniform(30, 120, n_samples),
        'airport_congestion_index': np.random.uniform(0, 1, n_samples),
        'weather_severity': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    def calculate_delay(row):
        prev_delay = row['prev_leg_arrival_delay_mins']
        buffer = row['scheduled_buffer_mins']
        congestion = row['airport_congestion_index']
        weather = row['weather_severity']
        
        # Calculate how much delay "spills over" into the buffer
        spillover = prev_delay - buffer
        
        if spillover <= 0:
            # Buffer fully absorbed the incoming delay.
            # Base operations are normal, but slight delays might occur independently 
            # due to severe weather + congestion affecting ground crew.
            target = 0
            if weather > 0 and congestion > 0.5:
                target += weather * 5 * congestion
        else:
            # Snowball effect triggered: the aircraft is arriving late enough to eat the buffer.
            # Congestion increases turnaround time friction.
            turnaround_penalty = 1.0 + (congestion * 1.5)
            
            # Weather directly slows down fueling, baggage, and boarding.
            weather_penalty = 1.0 + (weather * 0.5)
            
            # Base delayed departure
            target = spillover * turnaround_penalty * weather_penalty
            
            # Non-linear exponential compounding if conditions are completely terrible
            # High congestion and bad weather create a gridlock effect.
            if congestion > 0.8 and weather >= 1:
                target = target ** 1.15 
                
        # Introduce some Gaussian noise to simulate unobserved factors (maintenance, passengers)
        noise = np.random.normal(loc=0, scale=3)
        target += noise
        
        # Delay cannot physically be negative
        return max(0, target)

    df['target_departure_delay_mins'] = df.apply(calculate_delay, axis=1)
    return df

def main():
    print("1. Generating Synthetic Data...")
    df = generate_data()
    print(f"Dataset generated with shape: {df.shape}")
    print(df.head())
    print("-" * 50)
    
    # 2. XGBoost Model Pipeline
    print("2. Training XGBoost Baseline...")
    
    # Features for the model. 
    # tail_number is excluded in this baseline as it's a high-cardinality categorical
    # that requires specific encoding (like target encoding) to be useful without overfitting.
    features = ['prev_leg_arrival_delay_mins', 'scheduled_buffer_mins', 
                'airport_congestion_index', 'weather_severity']
    
    X = df[features]
    y = df['target_departure_delay_mins']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    # Ensure no negative predictions
    y_pred = np.maximum(0, y_pred)
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Evaluation Metrics:")
    print(f"RMSE: {rmse:.2f} mins")
    print(f"MAE:  {mae:.2f} mins")
    print("-" * 50)
    
    # 3. Explainability
    print("3. Generating Plots...")
    
    # Plot 1: Feature Importance
    plt.figure(figsize=(10, 6))
    
    # Use seaborn to create a nice barplot for feature importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('XGBoost Feature Importance (Gain)')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Saved 'feature_importance.png'")
    
    # Plot 2: Actual vs Predicted Scatter Plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='#1f77b4', edgecolor=None)
    
    # Diagonal reference line (Perfect prediction)
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Prediction')
    
    plt.xlabel('Actual Departure Delay (mins)')
    plt.ylabel('Predicted Departure Delay (mins)')
    plt.title('Actual vs Predicted Departure Delays')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    print("Saved 'actual_vs_predicted.png'")

if __name__ == "__main__":
    main()
