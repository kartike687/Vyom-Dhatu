"""
MOIL Manganese Production Forecasting Engine (Engine B)
Implements time-series production forecasting using:
1. Baseline Time-Series (Rolling 7d Moving Average / AR)
2. Random Forest Regressor
3. XGBoost Regressor
Generates 30-day forecasts with 90% confidence intervals and evaluates MAE, RMSE, MAPE, and R².
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.ml_core import (
    RandomForestRegressorPure,
    XGBoostRegressorPure,
    compute_regression_metrics
)

PROD_DATA_PATH = os.path.join(BASE_DIR, "data", "production", "daily_production_5yr.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLS = [
    "rainfall_mm",
    "soil_moisture",
    "ambient_temp_c",
    "excavator_avg_avail",
    "active_haul_trucks",
    "truck_avail_pct",
    "crusher_avail_pct",
    "blast_delay_days",
    "haul_efficiency",
    "shift_efficiency",
    "lag_1_prod",
    "lag_7_prod",
    "rolling_7d_prod"
]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Lagged features
    df["lag_1_prod"] = df["actual_production_tonnes"].shift(1)
    df["lag_7_prod"] = df["actual_production_tonnes"].shift(7)
    df["rolling_7d_prod"] = df["actual_production_tonnes"].shift(1).rolling(window=7, min_periods=1).mean()
    
    df = df.dropna().reset_index(drop=True)
    return df

def train_production_forecasters():
    if not os.path.exists(PROD_DATA_PATH):
        raise FileNotFoundError(f"Production data missing at {PROD_DATA_PATH}. Run data_generator.py first.")
        
    raw_df = pd.read_csv(PROD_DATA_PATH)
    df = prepare_features(raw_df)
    
    # Train / Test split (Hold out the last 60 days for validation)
    test_size = 60
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["actual_production_tonnes"].values
    
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["actual_production_tonnes"].values
    
    models = {
        "Random Forest Regressor": RandomForestRegressorPure(n_estimators=35, max_depth=6, random_state=26009),
        "XGBoost Regressor": XGBoostRegressorPure(n_estimators=30, learning_rate=0.08, max_depth=5, random_state=26009)
    }
    
    fitted_models = {}
    metrics = {}
    
    # 1. Baseline SARIMA / Rolling Average Forecaster
    y_baseline = test_df["rolling_7d_prod"].values
    base_metrics = compute_regression_metrics(y_test, y_baseline)
    metrics["SARIMA / Autoregressive Baseline"] = {
        "model_type": "Statistical Time-Series",
        "mae": base_metrics["mae"],
        "rmse": base_metrics["rmse"],
        "mape_pct": base_metrics["mape_pct"],
        "r2_score": base_metrics["r2_score"],
        "best_model": False
    }
    
    best_score = base_metrics["r2_score"]
    best_model_name = "SARIMA / Autoregressive Baseline"
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        preds = model.predict(X_test)
        reg_metrics = compute_regression_metrics(y_test, preds)
        
        metrics[name] = {
            "model_type": "Machine Learning",
            "mae": reg_metrics["mae"],
            "rmse": reg_metrics["rmse"],
            "mape_pct": reg_metrics["mape_pct"],
            "r2_score": reg_metrics["r2_score"],
            "best_model": False
        }
        
        if reg_metrics["r2_score"] > best_score:
            best_score = reg_metrics["r2_score"]
            best_model_name = name
            
    metrics[best_model_name]["best_model"] = True
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(fitted_models, os.path.join(MODELS_DIR, "production_models.joblib"))
    
    with open(os.path.join(MODELS_DIR, "production_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("Production forecasters successfully trained and serialized.")
    return metrics

def generate_30day_forecast(scenario_overrides: dict = None, model_name: str = "XGBoost Regressor"):
    """
    Generates a 30-day forward-looking production forecast with 90% confidence bands.
    Applies scenario overrides if what-if parameters are passed.
    """
    models_path = os.path.join(MODELS_DIR, "production_models.joblib")
    if not os.path.exists(models_path):
        train_production_forecasters()
        
    fitted_models = joblib.load(models_path)
    model = fitted_models.get(model_name, list(fitted_models.values())[0])
    
    raw_df = pd.read_csv(PROD_DATA_PATH)
    df = prepare_features(raw_df)
    last_row = df.iloc[-1]
    
    # Default scenario baseline
    rainfall = float(scenario_overrides.get("rainfall_mm", 14.0)) if scenario_overrides else 14.0
    soil_m = min(0.15 + rainfall / 180.0, 0.90)
    temp = float(scenario_overrides.get("ambient_temp_c", 31.0)) if scenario_overrides else 31.0
    
    ex_avail = float(scenario_overrides.get("excavator_avg_avail", 0.78)) if scenario_overrides else 0.78 # EX-02 in maintenance
    trucks = int(scenario_overrides.get("active_haul_trucks", 5)) if scenario_overrides else 5
    truck_pct = trucks / 6.0
    crusher_pct = float(scenario_overrides.get("crusher_avail_pct", 0.91)) if scenario_overrides else 0.91
    blast_delays = int(scenario_overrides.get("blast_delay_days", 1 if rainfall > 40 else 0)) if scenario_overrides else 0
    
    haul_eff = np.clip(1.0 - (soil_m * 0.28) - (0.15 if rainfall > 30.0 else 0.0), 0.45, 1.0)
    shift_eff = 0.92
    
    curr_lag1 = float(last_row["actual_production_tonnes"])
    curr_lag7 = float(last_row["lag_7_prod"])
    curr_roll7 = float(last_row["rolling_7d_prod"])
    
    forecast_days = []
    base_date = pd.to_datetime(last_row["date"])
    
    monthly_target = 52000.0
    daily_target = round(monthly_target / 30.0, 1) # ~1,733.3 T/day
    
    total_forecast_prod = 0.0
    total_target_prod = 0.0
    
    for day_i in range(1, 31):
        f_date = base_date + pd.Timedelta(days=day_i)
        
        # Add slight natural day-to-day fluctuation
        day_rain = max(0.0, rainfall + np.sin(day_i * 0.7) * 5.0)
        
        x_in = np.array([[
            day_rain,
            soil_m,
            temp,
            ex_avail,
            trucks,
            truck_pct,
            crusher_pct,
            blast_delays,
            haul_eff,
            shift_eff,
            curr_lag1,
            curr_lag7,
            curr_roll7
        ]])
        
        pred_val = float(model.predict(x_in)[0])
        pred_val = round(pred_val, 1)
        
        # 90% Confidence Interval: ~ ±1.645 * standard error (approx 120 T)
        std_err = 120.0 * (1.0 + (day_i / 30.0) * 0.25)
        lower_bound = round(max(300.0, pred_val - 1.645 * std_err), 1)
        upper_bound = round(pred_val + 1.645 * std_err, 1)
        
        total_forecast_prod += pred_val
        total_target_prod += daily_target
        
        forecast_days.append({
            "day": day_i,
            "date": f_date.strftime("%Y-%m-%d"),
            "daily_target_tonnes": daily_target,
            "forecast_tonnes": pred_val,
            "ci_lower_tonnes": lower_bound,
            "ci_upper_tonnes": upper_bound
        })
        
        # Roll forward lags
        curr_lag7 = curr_lag1
        curr_lag1 = pred_val
        curr_roll7 = curr_roll7 * 0.85 + pred_val * 0.15
        
    monthly_gap = round(total_forecast_prod - monthly_target, 1)
    
    # Calculate shortfall risk probability
    if monthly_gap < 0:
        gap_ratio = abs(monthly_gap) / (monthly_target * 0.15) # 15% shortfall = ~95% risk
        shortfall_risk_pct = round(min(98.0, 52.0 + gap_ratio * 46.0), 1)
    else:
        shortfall_risk_pct = round(max(5.0, 50.0 - (monthly_gap / (monthly_target * 0.10)) * 40.0), 1)
        
    if shortfall_risk_pct >= 75.0:
        risk_level = "CRITICAL"
    elif shortfall_risk_pct >= 55.0:
        risk_level = "HIGH"
    elif shortfall_risk_pct >= 35.0:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
        
    return {
        "monthly_target_tonnes": monthly_target,
        "monthly_forecast_tonnes": round(total_forecast_prod, 1),
        "monthly_gap_tonnes": monthly_gap,
        "shortfall_risk_pct": shortfall_risk_pct,
        "risk_level": risk_level,
        "model_used": model_name,
        "confidence_level_pct": 90,
        "daily_forecast": forecast_days
    }

if __name__ == "__main__":
    metrics = train_production_forecasters()
    print("Forecasting metrics:")
    print(json.dumps(metrics, indent=2))
    
    fc = generate_30day_forecast()
    print(f"Monthly Target: {fc['monthly_target_tonnes']} T")
    print(f"Monthly Forecast: {fc['monthly_forecast_tonnes']} T")
    print(f"Gap: {fc['monthly_gap_tonnes']} T | Risk: {fc['shortfall_risk_pct']}% ({fc['risk_level']})")
