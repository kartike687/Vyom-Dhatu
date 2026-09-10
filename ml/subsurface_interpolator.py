"""
MOIL Simulated Subsurface & Reserve Estimation Engine
Generates 3D block model voxels from synthetic drill holes and calculates resource tonnages
with Monte Carlo 90% uncertainty intervals.
Scientific Guardrail: Explicitly marked as SIMULATION DATA; not a statutory reserve code report.
"""

import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRILL_PATH = os.path.join(BASE_DIR, "data", "drilling", "drillholes.json")

def load_drillholes():
    if not os.path.exists(DRILL_PATH):
        raise FileNotFoundError(f"Drillholes data missing at {DRILL_PATH}")
    with open(DRILL_PATH, "r") as f:
        return json.load(f)

def compute_reserve_estimation(area_sq_m=280000.0, recovery_factor=0.82):
    """
    Standard mining resource estimation formula:
    Ore volume = Area x Average Thickness
    Estimated tonnage = Volume x Density
    Recoverable tonnage = Estimated tonnage x Recovery factor
    """
    drillholes = load_drillholes()
    ore_intercepts = [d["ore_thickness_m"] for d in drillholes if d["has_manganese_intercept"]]
    grades = [d["weighted_mn_grade_pct"] for d in drillholes if d["has_manganese_intercept"]]
    
    avg_thickness = float(np.mean(ore_intercepts)) if ore_intercepts else 16.5
    std_thickness = float(np.std(ore_intercepts)) if ore_intercepts else 3.8
    
    avg_grade = float(np.mean(grades)) if grades else 33.5
    std_grade = float(np.std(grades)) if grades else 4.2
    
    # Typical density for high-grade Sausar Group Braunite/Pyrolusite ore: 3.85 g/cm³ (t/m³)
    avg_density = 3.88
    
    # Monte Carlo simulation (1,000 iterations) for 90% confidence interval
    rng = np.random.RandomState(26009)
    sim_thickness = rng.normal(avg_thickness, std_thickness * 0.4, 1000)
    sim_density = rng.normal(avg_density, 0.12, 1000)
    sim_area = rng.normal(area_sq_m, area_sq_m * 0.05, 1000)
    
    sim_tonnage = (sim_area * sim_thickness * sim_density) / 1e6 # in Million Tonnes
    sim_recoverable = sim_tonnage * recovery_factor
    
    mean_geo_tonnage = float(np.mean(sim_tonnage))
    p5_geo = float(np.percentile(sim_tonnage, 5))
    p95_geo = float(np.percentile(sim_tonnage, 95))
    
    mean_rec_tonnage = float(np.mean(sim_recoverable))
    p5_rec = float(np.percentile(sim_recoverable, 5))
    p95_rec = float(np.percentile(sim_recoverable, 95))
    
    return {
        "orebody_surface_area_sq_m": round(area_sq_m, 0),
        "average_thickness_m": round(avg_thickness, 1),
        "average_density_t_m3": round(avg_density, 2),
        "average_mn_grade_pct": round(avg_grade, 1),
        "recovery_factor_pct": round(recovery_factor * 100, 1),
        "estimated_geological_tonnage_mt": round(mean_geo_tonnage, 2),
        "uncertainty_range_90_pct": {
            "lower_bound_mt": round(p5_geo, 2),
            "upper_bound_mt": round(p95_geo, 2)
        },
        "recoverable_tonnage_mt": round(mean_rec_tonnage, 2),
        "recoverable_uncertainty_range_90_pct": {
            "lower_bound_mt": round(p5_rec, 2),
            "upper_bound_mt": round(p95_rec, 2)
        },
        "disclaimer": "SIMULATION DATA — Educational / Demonstration Reserve Estimate. NOT statutory reserve data under UNFC/JORC codes."
    }

def generate_3d_block_model(nx=14, ny=14, nz=7):
    """
    Generates a 3D block model of voxels for interactive WebGL / Three.js visualization.
    X: 0 to 3200m (Easting)
    Y: 0 to 3200m (Northing)
    Z: 0 to 90m (Depth below collar)
    """
    drillholes = load_drillholes()
    
    # Collect ore intercept points for 3D distance interpolation
    ore_pts = []
    for d in drillholes:
        x, y, z_collar = d["easting_m"], d["northing_m"], d["collar_elev_m"]
        for it in d["intervals"]:
            mid_depth = (it["from_m"] + it["to_m"]) / 2.0
            ore_pts.append({
                "x": x,
                "y": y,
                "depth": mid_depth,
                "is_ore": it["is_ore"],
                "grade": it["mn_grade_pct"],
                "lithology": it["lithology"]
            })
            
    pts_arr = np.array([[p["x"], p["y"], p["depth"]] for p in ore_pts])
    grades_arr = np.array([p["grade"] for p in ore_pts])
    ore_flags = np.array([1.0 if p["is_ore"] else 0.0 for p in ore_pts])
    
    xs = np.linspace(400, 3000, nx)
    ys = np.linspace(800, 3000, ny)
    depths = np.linspace(5, 85, nz)
    
    voxels = []
    for x in xs:
        for y in ys:
            for d in depths:
                # 3D IDW interpolation
                dists = np.sqrt((pts_arr[:, 0] - x)**2 + (pts_arr[:, 1] - y)**2 + ((pts_arr[:, 2] - d) * 6.0)**2) # Anisotropy on Z
                weights = 1.0 / (np.maximum(dists, 20.0) ** 2.2)
                weights /= np.sum(weights)
                
                est_ore_prob = float(np.sum(weights * ore_flags))
                est_grade = float(np.sum(weights * grades_arr))
                
                # Assign stratigraphic unit
                if d < 18.0:
                    lithology = "Laterite Overburden"
                    grade = round(min(est_grade * 0.15, 4.0), 1)
                    color = "#92400e" # Amber brown
                    is_ore = False
                elif est_ore_prob >= 0.42 and 18.0 <= d <= 58.0:
                    lithology = "Manganiferous Reef"
                    grade = round(max(est_grade, 26.5), 1)
                    color = "#ec4899" if grade > 36.0 else "#8b5cf6" # Deep violet / magenta
                    is_ore = True
                else:
                    lithology = "Quartzite / Schist Waste"
                    grade = round(min(est_grade * 0.1, 2.5), 1)
                    color = "#475569" # Slate grey
                    is_ore = False
                    
                voxels.append({
                    "x": round(float(x), 1),
                    "y": round(float(y), 1),
                    "depth_m": round(float(d), 1),
                    "elevation_m": round(410.0 - float(d), 1),
                    "lithology": lithology,
                    "mn_grade_pct": grade,
                    "is_ore": is_ore,
                    "color": color
                })
                
    return {
        "grid_dimensions": {"nx": nx, "ny": ny, "nz": nz, "total_voxels": len(voxels)},
        "voxels": voxels,
        "drillhole_collars": [
            {
                "id": d["drill_id"],
                "x": d["easting_m"],
                "y": d["northing_m"],
                "elev_m": d["collar_elev_m"],
                "depth_m": d["total_depth_m"],
                "has_ore": d["has_manganese_intercept"],
                "grade": d["weighted_mn_grade_pct"]
            }
            for d in drillholes[:45] # First 45 collars for 3D viewport performance
        ]
    }

if __name__ == "__main__":
    res = compute_reserve_estimation()
    print("Reserve Estimation:")
    print(json.dumps(res, indent=2))
    
    bm = generate_3d_block_model(nx=8, ny=8, nz=4)
    print(f"Generated {bm['grid_dimensions']['total_voxels']} 3D block model voxels.")
