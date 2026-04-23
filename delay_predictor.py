# python/delay_predictor
# ML Model to Predict Bus Delays

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

MYSQL_PASSWORD = "chaitany41"

print("="*50)
print("STEP 1: Loading data from MySQL...")
engine = create_engine(
    f'mysql+pymysql://root:{MYSQL_PASSWORD}@localhost:3306/transport_analytics'
)

# Load sample of 500K rows (faster training)
df = pd.read_sql("""
    SELECT 
        route_id,
        direction,
        next_stop_name,
        delay_minutes,
        hour_of_day,
        day_of_week,
        is_peak_hour,
        month,
        distance_from_stop
    FROM trips
    LIMIT 500000
""", engine)

print(f"✅ Loaded {len(df):,} rows")

# ── STEP 2: Feature Engineering ──
print("\nSTEP 2: Preparing features...")

# Encode categorical columns
le_route = LabelEncoder()
le_day   = LabelEncoder()
le_stop  = LabelEncoder()

df['route_encoded'] = le_route.fit_transform(
    df['route_id'].astype(str)
)
df['day_encoded'] = le_day.fit_transform(
    df['day_of_week'].astype(str)
)
df['stop_encoded'] = le_stop.fit_transform(
    df['next_stop_name'].fillna('Unknown').astype(str)
)

# Clean distance column
df['distance_from_stop'] = pd.to_numeric(
    df['distance_from_stop'], errors='coerce'
).fillna(0)

# Final features
features = [
    'route_encoded',
    'day_encoded',
    'hour_of_day',
    'is_peak_hour',
    'stop_encoded',
    'month',
    'distance_from_stop'
]

X = df[features].fillna(0)
y = df['delay_minutes']

print(f"✅ Features ready: {features}")
print(f"   Training on {len(X):,} samples")

# ── STEP 3: Train Model ──
print("\nSTEP 3: Training Random Forest model...")
print("(This takes 2-3 minutes — please wait...)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    random_state=42,
    n_jobs=-1       # use all CPU cores
)
model.fit(X_train, y_train)
print("✅ Model trained!")

# ── STEP 4: Evaluate ──
print("\nSTEP 4: Evaluating model...")
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / 
               (y_test + 0.001))) * 100

print("\n" + "="*40)
print("📊 MODEL PERFORMANCE:")
print("="*40)
print(f"  MAE  (Mean Abs Error) : {mae:.2f} minutes")
print(f"  R²   (Accuracy Score) : {r2:.3f}")
print(f"  MAPE (% Error)        : {mape:.1f}%")
print("="*40)

# ── STEP 5: Feature Importance ──
print("\nSTEP 5: Feature Importance...")
importance_df = pd.DataFrame({
    'feature':    features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nWhat drives delays the most?")
for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:25} {bar} {row['importance']:.3f}")

# ── STEP 6: Save Results ──
print("\nSTEP 6: Saving results...")
BASE = r'C:\Users\Mane Chaitanya\Desktop\transport-delay-project\data\processed'

# Save feature importance
importance_df.to_csv(
    f'{BASE}\\feature_importance.csv',
    index=False
)

# Save predictions sample
results = pd.DataFrame({
    'actual_delay':    y_test.values[:1000],
    'predicted_delay': y_pred[:1000],
    'error_minutes':   abs(y_test.values[:1000] - y_pred[:1000])
})
results.to_csv(
    f'{BASE}\\model_predictions.csv',
    index=False
)

# ── STEP 7: Plot Feature Importance ──
plt.figure(figsize=(10, 6))
plt.barh(
    importance_df['feature'],
    importance_df['importance'],
    color=['#e74c3c' if i == 0 else '#3498db' 
           for i in range(len(importance_df))]
)
plt.xlabel('Importance Score')
plt.title('What Causes Bus Delays?\n(Random Forest Feature Importance)',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(
    f'{BASE}\\feature_importance.png',
    dpi=150, bbox_inches='tight'
)
plt.show()

# ── STEP 8: Test Live Prediction ──
print("\nSTEP 7: Live prediction demo...")
print("\nPredicting delay for:")
print("  Route: S86 | Friday | 6PM | Peak Hour")

sample = pd.DataFrame([{
    'route_encoded':      le_route.transform(['S86'])[0] 
                          if 'S86' in le_route.classes_ else 0,
    'day_encoded':        le_day.transform(['Friday'])[0],
    'hour_of_day':        18,
    'is_peak_hour':       1,
    'stop_encoded':       0,
    'month':              12,
    'distance_from_stop': 200
}])

predicted = model.predict(sample)[0]
print(f"\n  🚌 Predicted Delay: {predicted:.1f} minutes")
print(f"  {'🔴 HIGH DELAY!' if predicted > 10 else '🟡 MODERATE' if predicted > 5 else '🟢 LOW'}")

print("\n" + "="*50)
print("🎉 ML MODEL COMPLETE!")
print("="*50)
print(f"\nFiles saved:")
print(f"  ✅ feature_importance.csv")
print(f"  ✅ feature_importance.png")
print(f"  ✅ model_predictions.csv")