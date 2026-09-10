"""
MOIL Manganese Mine Intelligence & Digital Twin Simulator
Synthetic Data Generator
Seed: RANDOM_SEED = 26009 for 100% reproducibility.
Scientific Guardrail: Environmental and satellite variables (NDVI, rainfall, soil moisture, LST)
are categorized as operational/environmental risk proxies and surface alteration indicators,
NOT direct underground manganese detectors.
"""

import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 26009
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def ensure_directories():
    for sub in ["raw", "processed", "geological", "drilling", "satellite", "weather", "production", "equipment", "simulation"]:
        os.makedirs(os.path.join(DATA_DIR, sub), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

def generate_geological_zones():
    """
    Generates 6 distinct exploration zones inspired by the Balaghat-Ukwa-Dongri Buzurg manganese belt
    in the Sausar Group (Central India). Coordinates approximately 21.85°N, 79.82°E.
    """
    zones = [
        {
            "id": "ZONE_A",
            "name": "Zone A - North Balaghat Extension",
            "center": [21.8820, 79.8350],
            "area_sq_km": 4.8,
            "lithology": "Mansar Schist / Gondite Formation",
            "structural_density": 0.88,       # Faults & shear zones (0-1)
            "lineament_density": 0.84,
            "spectral_alteration_index": 0.82, # Clay/OH & iron oxide alteration
            "distance_to_known_deposit_km": 1.2,
            "elevation_m": 412.0,
            "slope_deg": 14.5,
            "drainage_density": 0.65,
            "terrain_ruggedness": 0.58,
            "historical_prospect_points": 14,
            "known_manganese_occurrences": True,
            "true_prospectivity_prob": 0.92,  # Ground truth probability
            "prospectivity_tier": "VERY HIGH",
            "recommended_action": "Prioritize geological verification / intensive core drilling."
        },
        {
            "id": "ZONE_B",
            "name": "Zone B - South Ukwa Limb",
            "center": [21.8450, 79.8620],
            "area_sq_km": 6.2,
            "lithology": "Quartz-Muscovite Schist with Gondite bands",
            "structural_density": 0.68,
            "lineament_density": 0.71,
            "spectral_alteration_index": 0.64,
            "distance_to_known_deposit_km": 3.4,
            "elevation_m": 385.0,
            "slope_deg": 11.2,
            "drainage_density": 0.52,
            "terrain_ruggedness": 0.44,
            "historical_prospect_points": 8,
            "known_manganese_occurrences": True,
            "true_prospectivity_prob": 0.68,
            "prospectivity_tier": "MEDIUM",
            "recommended_action": "Conduct infill geophysical resistivity and trench sampling."
        },
        {
            "id": "ZONE_C",
            "name": "Zone C - Tirodi Gneissic Border",
            "center": [21.8100, 79.7900],
            "area_sq_km": 5.1,
            "lithology": "Biotite Gneiss / Amphibolite Basement",
            "structural_density": 0.28,
            "lineament_density": 0.32,
            "spectral_alteration_index": 0.22,
            "distance_to_known_deposit_km": 9.1,
            "elevation_m": 330.0,
            "slope_deg": 5.4,
            "drainage_density": 0.38,
            "terrain_ruggedness": 0.25,
            "historical_prospect_points": 1,
            "known_manganese_occurrences": False,
            "true_prospectivity_prob": 0.22,
            "prospectivity_tier": "LOW",
            "recommended_action": "Low exploration priority; standard regional reconnaissance."
        },
        {
            "id": "ZONE_D",
            "name": "Zone D - Dongri Buzurg East Shear",
            "center": [21.8650, 79.8050],
            "area_sq_km": 5.5,
            "lithology": "Gondite & Secondary Supergene Enriched Pyrolusite",
            "structural_density": 0.91,
            "lineament_density": 0.89,
            "spectral_alteration_index": 0.86,
            "distance_to_known_deposit_km": 1.6,
            "elevation_m": 425.0,
            "slope_deg": 16.2,
            "drainage_density": 0.70,
            "terrain_ruggedness": 0.62,
            "historical_prospect_points": 12,
            "known_manganese_occurrences": True,
            "true_prospectivity_prob": 0.89,
            "prospectivity_tier": "VERY HIGH",
            "recommended_action": "Fast-track exploratory borehole drilling and assay evaluation."
        },
        {
            "id": "ZONE_E",
            "name": "Zone E - Bharweli South Deep Syncline",
            "center": [21.8300, 79.8300],
            "area_sq_km": 7.0,
            "lithology": "Mansar Phyllite & Manganiferous Quartzite",
            "structural_density": 0.62,
            "lineament_density": 0.58,
            "spectral_alteration_index": 0.55,
            "distance_to_known_deposit_km": 4.1,
            "elevation_m": 365.0,
            "slope_deg": 9.8,
            "drainage_density": 0.48,
            "terrain_ruggedness": 0.39,
            "historical_prospect_points": 6,
            "known_manganese_occurrences": True,
            "true_prospectivity_prob": 0.59,
            "prospectivity_tier": "MODERATE",
            "recommended_action": "Evaluate subsurface structural plunge via 2D seismic/IP survey."
        },
        {
            "id": "ZONE_F",
            "name": "Zone F - Alluvial Sedimentary Basin",
            "center": [21.7850, 79.8500],
            "area_sq_km": 8.4,
            "lithology": "Recent Alluvium / Deccan Trap Overburden",
            "structural_density": 0.15,
            "lineament_density": 0.18,
            "spectral_alteration_index": 0.16,
            "distance_to_known_deposit_km": 12.5,
            "elevation_m": 310.0,
            "slope_deg": 3.1,
            "drainage_density": 0.30,
            "terrain_ruggedness": 0.15,
            "historical_prospect_points": 0,
            "known_manganese_occurrences": False,
            "true_prospectivity_prob": 0.11,
            "prospectivity_tier": "LOW",
            "recommended_action": "Unfavorable stratigraphy for open-cast or shallow manganese."
        }
    ]

    path = os.path.join(DATA_DIR, "geological", "geological_zones.json")
    with open(path, "w") as f:
        json.dump(zones, f, indent=2)
    return zones

def generate_spatial_prospectivity_grid(num_cells=400):
    """
    Generates a 20x20 regular spatial raster across the 15x15 km exploration perimeter.
    Calculates geological features + satellite environmental risk features.
    """
    np.random.seed(RANDOM_SEED)
    grid_records = []
    
    # Grid limits around Balaghat Mine coordinates
    lat_min, lat_max = 21.78, 21.90
    lon_min, lon_max = 79.76, 79.90
    
    lats = np.linspace(lat_min, lat_max, int(np.sqrt(num_cells)))
    lons = np.linspace(lon_min, lon_max, int(np.sqrt(num_cells)))
    
    # Centers of known high mineralization
    known_centers = [
        (21.8820, 79.8350, 0.95),  # Balaghat north
        (21.8650, 79.8050, 0.90),  # Dongri Buzurg East
        (21.8450, 79.8620, 0.70),  # Ukwa
    ]
    
    cell_id = 1
    for lat in lats:
        for lon in lons:
            # Distance to nearest known manganese deposit in km
            dists = [math.sqrt((lat - clat)**2 + (lon - clon)**2) * 111.0 for clat, clon, _ in known_centers]
            min_dist = min(dists)
            nearest_idx = dists.index(min_dist)
            influence = known_centers[nearest_idx][2] * math.exp(-min_dist / 3.2)
            
            # Geological features (causal to subsurface manganese)
            structural_density = np.clip(influence * 0.85 + np.random.normal(0.25, 0.10), 0.05, 0.98)
            lineament_density = np.clip(structural_density * 0.9 + np.random.normal(0.05, 0.08), 0.05, 0.98)
            lithology_score = np.clip(influence * 0.90 + np.random.normal(0.20, 0.12), 0.02, 0.99)
            spectral_alteration = np.clip(influence * 0.75 + np.random.normal(0.25, 0.09), 0.05, 0.95)
            elevation = 310.0 + (lat - lat_min)/(lat_max - lat_min) * 120.0 + np.random.normal(0, 10.0)
            slope = np.clip(5.0 + structural_density * 14.0 + np.random.normal(0, 2.0), 1.0, 28.0)
            
            # Satellite / Environmental features (explicitly NOT subsurface detectors)
            # Proxy for surface vegetation / moisture / operational constraints
            ndvi = np.clip(0.18 + np.sin(lon * 20.0) * 0.12 + np.random.normal(0.15, 0.05), 0.05, 0.75)
            ndwi = np.clip(-0.15 + (1.0 - slope/30.0)*0.20 + np.random.normal(0, 0.04), -0.40, 0.40)
            lst_celsius = np.clip(38.0 - (elevation - 310.0)*0.04 + np.random.normal(0, 1.5), 24.0, 44.0)
            soil_moisture = np.clip(0.15 + ndwi*0.4 + np.random.normal(0.15, 0.04), 0.05, 0.85)
            bare_soil_index = np.clip(1.0 - ndvi + np.random.normal(0, 0.05), 0.10, 0.95)
            
            # True prospectivity probability (driven strictly by geological features, not vegetation/rain)
            log_odds = (
                3.2 * structural_density +
                3.5 * lithology_score +
                2.8 * spectral_alteration +
                2.0 * lineament_density -
                0.35 * min_dist +
                0.01 * (elevation - 350.0) -
                4.2
            )
            prob = 1.0 / (1.0 + math.exp(-log_odds))
            prob = np.clip(prob, 0.03, 0.97)
            
            # Binary label for classification training
            mineralized = 1 if prob >= 0.50 else 0
            
            grid_records.append({
                "cell_id": f"CELL_{cell_id:04d}",
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "min_distance_to_known_deposit_km": round(min_dist, 2),
                "structural_density": round(structural_density, 3),
                "lineament_density": round(lineament_density, 3),
                "lithology_favorability": round(lithology_score, 3),
                "spectral_alteration_index": round(spectral_alteration, 3),
                "elevation_m": round(elevation, 1),
                "slope_deg": round(slope, 1),
                # Environmental risk features
                "ndvi": round(ndvi, 3),
                "ndwi": round(ndwi, 3),
                "land_surface_temp_c": round(lst_celsius, 1),
                "soil_moisture": round(soil_moisture, 3),
                "bare_soil_index": round(bare_soil_index, 3),
                # Target
                "ground_truth_probability": round(prob, 3),
                "mineralized_target": mineralized
            })
            cell_id += 1
            
    df = pd.DataFrame(grid_records)
    df.to_csv(os.path.join(DATA_DIR, "geological", "prospectivity_grid.csv"), index=False)
    return df

def generate_drillhole_dataset(num_drillholes=125):
    """
    Generates synthetic drill hole intercepts with stratigraphic logging, assays, and coordinates.
    Stratigraphy based on Sausar Group: Laterite overburden -> Manganiferous gondite/reef -> Quartzite/Gneiss waste.
    """
    np.random.seed(RANDOM_SEED)
    drillholes = []
    
    # 4 clusters of drill holes corresponding to active & prospective pits
    cluster_centers = [
        {"x": 1200, "y": 2400, "elev": 415, "name": "Balaghat North Pit", "prob_ore": 0.85},
        {"x": 2800, "y": 1800, "elev": 395, "name": "Ukwa Deep Extension", "prob_ore": 0.70},
        {"x": 800,  "y": 1100, "elev": 425, "name": "Dongri Buzurg East", "prob_ore": 0.88},
        {"x": 2100, "y": 3100, "elev": 360, "name": "Exploration Outlier", "prob_ore": 0.25}
    ]
    
    for i in range(1, num_drillholes + 1):
        cluster = cluster_centers[np.random.choice(len(cluster_centers), p=[0.40, 0.25, 0.25, 0.10])]
        x = cluster["x"] + np.random.normal(0, 220)
        y = cluster["y"] + np.random.normal(0, 220)
        z = cluster["elev"] + np.random.normal(0, 4)
        
        total_depth = np.random.choice([60.0, 75.0, 90.0, 110.0, 130.0], p=[0.2, 0.3, 0.3, 0.15, 0.05])
        
        # Stratigraphic intervals
        overburden_thickness = np.random.uniform(12.0, 26.0)
        has_ore = np.random.rand() < cluster["prob_ore"]
        
        intervals = []
        # 1. Overburden
        intervals.append({
            "from_m": 0.0,
            "to_m": round(overburden_thickness, 1),
            "lithology": "Laterite / Alluvial Gravel",
            "mn_grade_pct": round(np.random.uniform(1.2, 4.5), 2),
            "density_g_cm3": 2.25,
            "is_ore": False
        })
        
        curr_depth = overburden_thickness
        if has_ore:
            ore_thickness = np.random.uniform(8.0, 28.0)
            ore_to = min(curr_depth + ore_thickness, total_depth - 10.0)
            # Manganese grade distribution (typically 28% to 46% Mn for high grade MOIL ore)
            mn_grade = np.random.normal(34.2, 4.8)
            mn_grade = np.clip(mn_grade, 22.0, 48.5)
            density = round(3.65 + (mn_grade - 22.0)/26.5 * 0.65, 2) # Heavier manganese oxides
            
            intervals.append({
                "from_m": round(curr_depth, 1),
                "to_m": round(ore_to, 1),
                "lithology": "Manganiferous Reef (Braunzite / Pyrolusite)",
                "mn_grade_pct": round(mn_grade, 2),
                "density_g_cm3": density,
                "is_ore": True
            })
            curr_depth = ore_to
            
        # Footwall waste rock
        intervals.append({
            "from_m": round(curr_depth, 1),
            "to_m": round(total_depth, 1),
            "lithology": "Mansar Mica Schist / Quartzite",
            "mn_grade_pct": round(np.random.uniform(0.5, 3.2), 2),
            "density_g_cm3": 2.70,
            "is_ore": False
        })
        
        # Summary metrics for drillhole header
        ore_intervals = [it for it in intervals if it["is_ore"]]
        tot_ore_len = sum(it["to_m"] - it["from_m"] for it in ore_intervals)
        avg_grade = np.average([it["mn_grade_pct"] for it in ore_intervals], weights=[it["to_m"] - it["from_m"] for it in ore_intervals]) if ore_intervals else 0.0
        
        drillholes.append({
            "drill_id": f"DRILL-{i:03d}",
            "easting_m": round(x, 1),
            "northing_m": round(y, 1),
            "collar_elev_m": round(z, 1),
            "total_depth_m": round(total_depth, 1),
            "cluster_area": cluster["name"],
            "has_manganese_intercept": has_ore,
            "ore_thickness_m": round(tot_ore_len, 1),
            "weighted_mn_grade_pct": round(avg_grade, 2),
            "intervals": intervals
        })
        
    path = os.path.join(DATA_DIR, "drilling", "drillholes.json")
    with open(path, "w") as f:
        json.dump(drillholes, f, indent=2)
    return drillholes

def generate_fleet_and_operational_records():
    """
    Generates 5 years (1,825 days) of daily operational logs, plus real-time fleet digital twin state.
    Captures causal relationships:
    - High rainfall -> lower haul speed -> increased slip/downtime -> lower production
    - Excavator breakdown (EX-02) -> immediate shovel bottleneck
    - Blast delay -> insufficient muckpile ore availability
    - Crusher maintenance -> stockpile choking
    """
    np.random.seed(RANDOM_SEED)
    start_date = datetime(2021, 1, 1)
    num_days = 1826 # 5 years up to 2026
    
    daily_records = []
    base_daily_target = 1733.3 # ~52,000 tonnes per 30-day month
    
    for d in range(num_days):
        current_date = start_date + timedelta(days=d)
        month = current_date.month
        
        # Monsoon seasonality for Central India (July-September heavy rainfall)
        is_monsoon = month in [6, 7, 8, 9]
        if is_monsoon:
            rainfall_mm = np.random.exponential(scale=28.0) if np.random.rand() < 0.65 else np.random.uniform(0, 5)
            ambient_temp = np.random.normal(28.0, 3.0)
            soil_moisture = np.clip(0.35 + rainfall_mm / 140.0, 0.20, 0.95)
        else:
            rainfall_mm = np.random.exponential(scale=2.0) if np.random.rand() < 0.12 else 0.0
            ambient_temp = np.random.normal(36.0, 5.0) if month in [4, 5] else np.random.normal(24.0, 4.0)
            soil_moisture = np.clip(0.12 + rainfall_mm / 200.0, 0.05, 0.35)
            
        # Equipment fleet configuration
        # 2 Heavy Hydraulic Excavators (EX-01, EX-02)
        ex1_avail = 0.92 if np.random.rand() > 0.08 else np.random.uniform(0.4, 0.7)
        ex2_avail = 0.90 if np.random.rand() > 0.12 else np.random.uniform(0.3, 0.65)
        excavator_avail_pct = (ex1_avail + ex2_avail) / 2.0
        
        # 6 Haul Trucks (HT-01 to HT-06)
        active_trucks = np.random.choice([4, 5, 6], p=[0.15, 0.45, 0.40])
        truck_avail_pct = active_trucks / 6.0
        
        # Primary In-Pit Crusher (CR-01)
        crusher_avail_pct = 0.94 if np.random.rand() > 0.07 else np.random.uniform(0.50, 0.75)
        
        # Blasting operations
        blast_delay_days = 0
        if rainfall_mm > 45.0:
            blast_delay_days = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        elif np.random.rand() < 0.05:
            blast_delay_days = 1
            
        # Haul road condition factor
        haul_efficiency = 1.0 - (soil_moisture * 0.28) - (0.15 if rainfall_mm > 30.0 else 0.0)
        haul_efficiency = np.clip(haul_efficiency, 0.45, 1.0)
        
        # Shift efficiency
        shift_eff = np.random.normal(0.92, 0.04)
        if ambient_temp > 42.0:
            shift_eff -= 0.08 # Heat stress impact
        shift_eff = np.clip(shift_eff, 0.65, 1.0)
        
        # Production simulation with physical bottlenecks
        # Maximum capacity governed by Leontief-style min bottleneck
        shovel_capacity = (ex1_avail * 1050.0 + ex2_avail * 950.0)
        haulage_capacity = active_trucks * 360.0 * haul_efficiency
        crusher_capacity = 2100.0 * crusher_avail_pct
        blast_factor = 1.0 if blast_delay_days == 0 else max(0.40, 1.0 - blast_delay_days * 0.25)
        
        potential_tonnage = min(shovel_capacity, haulage_capacity, crusher_capacity) * blast_factor * shift_eff
        actual_production = np.clip(potential_tonnage + np.random.normal(0, 45.0), 300.0, 2400.0)
        
        target_tonnage = base_daily_target + (50.0 if not is_monsoon else -50.0)
        shortfall_gap = actual_production - target_tonnage
        is_shortfall = 1 if shortfall_gap < -150.0 else 0
        
        # Satellite features logged weekly/interpolated daily
        ndvi = np.clip(0.35 + (0.25 if is_monsoon else -0.10) + np.random.normal(0, 0.02), 0.12, 0.72)
        ndwi = np.clip(-0.10 + (soil_moisture * 0.4) + np.random.normal(0, 0.02), -0.30, 0.35)
        lst_c = np.clip(ambient_temp + 3.5 - (soil_moisture * 5.0), 20.0, 48.0)
        
        daily_records.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "target_tonnage": round(target_tonnage, 1),
            "actual_production_tonnes": round(actual_production, 1),
            "shortfall_gap_tonnes": round(shortfall_gap, 1),
            "is_shortfall": is_shortfall,
            # Weather & Environment
            "rainfall_mm": round(rainfall_mm, 1),
            "soil_moisture": round(soil_moisture, 3),
            "ambient_temp_c": round(ambient_temp, 1),
            "ndvi": round(ndvi, 3),
            "ndwi": round(ndwi, 3),
            "land_surface_temp_c": round(lst_c, 1),
            # Equipment & Operations
            "excavator_ex01_avail": round(ex1_avail, 2),
            "excavator_ex02_avail": round(ex2_avail, 2),
            "excavator_avg_avail": round(excavator_avail_pct, 2),
            "active_haul_trucks": int(active_trucks),
            "truck_avail_pct": round(truck_avail_pct, 2),
            "crusher_avail_pct": round(crusher_avail_pct, 2),
            "blast_delay_days": int(blast_delay_days),
            "haul_efficiency": round(haul_efficiency, 2),
            "shift_efficiency": round(shift_eff, 2)
        })
        
    df_prod = pd.DataFrame(daily_records)
    df_prod.to_csv(os.path.join(DATA_DIR, "production", "daily_production_5yr.csv"), index=False)
    
    # Live Fleet Status Digital Twin snapshot
    fleet = [
        {"id": "EX-01", "name": "Excavator EX-01", "type": "Hydraulic Excavator (5.5m³)", "zone": "Zone A Pit", "availability_pct": 92.0, "status": "RUNNING", "engine_temp_c": 86.4, "hours_run_today": 16.4, "next_pm_days": 12, "x": 1250, "y": 2420},
        {"id": "EX-02", "name": "Excavator EX-02", "type": "Hydraulic Excavator (4.2m³)", "zone": "Zone D East", "availability_pct": 64.0, "status": "MAINTENANCE", "engine_temp_c": 104.2, "hours_run_today": 6.1, "next_pm_days": 0, "x": 820, "y": 1140},
        {"id": "HT-01", "name": "Haul Truck HT-01", "type": "60T Off-Highway Dumper", "zone": "Haul Road 1", "availability_pct": 88.0, "status": "RUNNING", "engine_temp_c": 82.0, "hours_run_today": 17.1, "next_pm_days": 18, "x": 1420, "y": 2210},
        {"id": "HT-02", "name": "Haul Truck HT-02", "type": "60T Off-Highway Dumper", "zone": "Pit Bench 3", "availability_pct": 71.0, "status": "WARNING", "engine_temp_c": 96.5, "hours_run_today": 13.5, "next_pm_days": 2, "x": 1180, "y": 2390},
        {"id": "HT-03", "name": "Haul Truck HT-03", "type": "60T Off-Highway Dumper", "zone": "Crusher Loop", "availability_pct": 95.0, "status": "RUNNING", "engine_temp_c": 80.2, "hours_run_today": 18.0, "next_pm_days": 24, "x": 1850, "y": 1750},
        {"id": "HT-04", "name": "Haul Truck HT-04", "type": "60T Off-Highway Dumper", "zone": "Zone B Haulage", "availability_pct": 89.0, "status": "RUNNING", "engine_temp_c": 83.1, "hours_run_today": 16.8, "next_pm_days": 15, "x": 2650, "y": 1820},
        {"id": "HT-05", "name": "Haul Truck HT-05", "type": "60T Off-Highway Dumper", "zone": "Workshop", "availability_pct": 52.0, "status": "MAINTENANCE", "engine_temp_c": 45.0, "hours_run_today": 2.0, "next_pm_days": 0, "x": 2100, "y": 1600},
        {"id": "HT-06", "name": "Haul Truck HT-06", "type": "60T Off-Highway Dumper", "zone": "Stockpile 2", "availability_pct": 91.0, "status": "RUNNING", "engine_temp_c": 81.5, "hours_run_today": 17.5, "next_pm_days": 20, "x": 1950, "y": 1680},
        {"id": "DR-01", "name": "Drill Rig DR-01", "type": "Rotary Blast-hole Drill (150mm)", "zone": "Zone A Bench", "availability_pct": 84.0, "status": "RUNNING", "engine_temp_c": 85.0, "hours_run_today": 14.2, "next_pm_days": 8, "x": 1310, "y": 2480},
        {"id": "DR-02", "name": "Drill Rig DR-02", "type": "Core Exploration Rig", "zone": "Zone B Extension", "availability_pct": 78.0, "status": "RUNNING", "engine_temp_c": 79.4, "hours_run_today": 12.0, "next_pm_days": 9, "x": 2780, "y": 1890},
        {"id": "CR-01", "name": "Crusher CR-01", "type": "Primary Jaw Crusher (350 TPH)", "zone": "Plant Central", "availability_pct": 91.0, "status": "RUNNING", "engine_temp_c": 78.5, "hours_run_today": 19.5, "next_pm_days": 11, "x": 1900, "y": 1700}
    ]
    
    with open(os.path.join(DATA_DIR, "equipment", "fleet_status.json"), "w") as f:
        json.dump(fleet, f, indent=2)
        
    return df_prod, fleet

def run_all_generators():
    print("Generating MOIL Manganese Digital Twin synthetic dataset (Seed: 26009)...")
    ensure_directories()
    zones = generate_geological_zones()
    print(f" -> Generated {len(zones)} geological zones.")
    grid_df = generate_spatial_prospectivity_grid(num_cells=400)
    print(f" -> Generated {len(grid_df)} prospectivity raster cells.")
    drillholes = generate_drillhole_dataset(num_drillholes=125)
    print(f" -> Generated {len(drillholes)} drill holes with 3D intervals & Mn assay.")
    df_prod, fleet = generate_fleet_and_operational_records()
    print(f" -> Generated {len(df_prod)} daily production records (5 years) & {len(fleet)} fleet digital twin nodes.")
    print("Dataset generation complete!")

if __name__ == "__main__":
    run_all_generators()
