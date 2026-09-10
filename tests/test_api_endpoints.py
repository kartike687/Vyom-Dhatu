"""
Automated Integration & Scenario Tests for MOIL Manganese Mine Intelligence Platform
Direct functional unit tests verifying ML models, GIS logic, What-If simulation,
and optimization routines without external HTTP client dependencies.
"""

import sys
import os
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app.main import (
    get_dashboard_summary,
    get_prospectivity,
    simulate_drill_hole,
    get_subsurface_block_model,
    get_production_forecast,
    get_fleet_telemetry,
    run_what_if_simulation,
    apply_optimization_interventions,
    toggle_fallback_mode,
    DrillSimulateRequest,
    WhatIfRequest
)

def test_dashboard_summary():
    data = get_dashboard_summary()
    assert "kpis" in data
    assert data["kpis"]["monthly_target_tonnes"] == 52000.0
    assert "SIMULATION MODE" in data["operating_mode"]
    assert len(data["data_sources"]) >= 4

def test_prospectivity_grid_and_models():
    data = get_prospectivity(model="XGBoost")
    assert len(data["zones"]) == 6
    zone_a = next(z for z in data["zones"] if z["id"] == "ZONE_A")
    assert zone_a["ai_priority"] in ["HIGH", "VERY HIGH"]
    assert "roc_auc" in data["model_performance"]
    assert data["model_performance"]["roc_auc"] >= 0.90

def test_drill_simulation():
    req = DrillSimulateRequest(
        zone_id="ZONE_A",
        latitude=21.8820,
        longitude=79.8350,
        target_depth_m=80.0
    )
    data = simulate_drill_hole(req)
    assert data["potential_mn_bearing_interval_m"] > 0
    assert data["estimated_average_grade_pct"] >= 25.0
    assert len(data["intervals"]) >= 3

def test_subsurface_block_model():
    data = get_subsurface_block_model()
    assert "reserve_estimation" in data
    assert data["reserve_estimation"]["estimated_geological_tonnage_mt"] > 0
    assert "block_model" in data
    assert len(data["block_model"]["voxels"]) > 0

def test_production_forecast():
    data = get_production_forecast(model="XGBoost Regressor")
    assert len(data["daily_forecast"]) == 30
    assert "all_models_metrics" in data
    assert data["monthly_target_tonnes"] == 52000.0

def test_fleet_telemetry():
    data = get_fleet_telemetry()
    assert data["summary"]["total_machines"] == 11
    assert any(m["id"] == "EX-01" for m in data["fleet_nodes"])

def test_what_if_and_optimization():
    req = WhatIfRequest(
        rainfall_mm=190.0,
        excavator_avg_avail=0.65,
        active_haul_trucks=4,
        crusher_avail_pct=0.88,
        blast_delay_days=2,
        model_name="XGBoost Regressor"
    )
    sim_data = run_what_if_simulation(req)
    assert sim_data["production_gap_tonnes"] < 0
    assert sim_data["shortfall_risk_pct"] > 70.0
    assert len(sim_data["root_cause_analysis"]["contributors"]) >= 4

    # Apply Optimization
    opt_data = apply_optimization_interventions()
    assert opt_data["status"] == "SUCCESS"
    assert opt_data["revised_shortfall_risk_pct"] < sim_data["shortfall_risk_pct"]

def test_fallback_toggle():
    data1 = toggle_fallback_mode()
    assert data1["confidence_score"] == 61.0
    assert data1["missing_data_fallback_active"] is True

    # Toggle back
    data2 = toggle_fallback_mode()
    assert data2["confidence_score"] == 82.0
    assert data2["missing_data_fallback_active"] is False

if __name__ == "__main__":
    pytest.main(["-v", __file__])
