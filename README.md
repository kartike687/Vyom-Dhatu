# MOIL Manganese Mine Intelligence & Digital Twin Simulator
### Smart India Hackathon (SIH) 2026 Problem Statement Solution
> **"Using AI/ML and Space Technology to Identify Manganese Reserves and Overcome Production Shortfalls."**  
> *Targeted for MOIL Limited (Balaghat, Ukwa, Dongri Buzurg, Tirodi Manganese Belt — Central India)*

---

## 1. Executive Summary & Problem Overview
India's largest manganese ore producer, **MOIL Limited**, operates underground and opencast mines across Madhya Pradesh and Maharashtra (Sausar Geological Group). Meeting surging industrial demand for manganese in steelmaking and lithium-manganese EV batteries requires:
1. **Accelerated Exploration Intelligence**: Shifting from slow, trial-and-error drilling to predictive geological and remote-sensing prospectivity targeting.
2. **Operational Resilience & Shortfall Mitigation**: Overcoming severe production losses caused by seasonal monsoons (July–September), pit road slickness, shovel/excavator breakdowns, and blasting delays.

This platform provides an integrated, offline-first **AI-Powered Mining Command Centre & Digital Twin Simulator** that combines:
- **Engine A**: AI Mineral Prospectivity & 3D Subsurface Resource Intelligence.
- **Engine B**: Production Shortfall Forecasting, Root-Cause Explainability (TreeSHAP), and Operational Constraint Optimization.
- **Space Intelligence**: Sentinel-2 & ISRO Bhuvan satellite indices (NDVI, NDWI, LST, Soil Moisture, Rainfall) modeling operational weather risks.
- **Equipment Digital Twin**: Real-time telemetry, animated material flow pipeline, and failure injection simulator.
- **SIH 2026 Judge Mode**: A 1-click automated 9-step Hackathon presentation flow.

---

## 2. Scientific Guardrails & Regulatory Compliance
To ensure complete geological and engineering integrity:
> [!IMPORTANT]
> **No Direct Detection Fallacy**: Environmental and satellite features (rainfall, soil moisture, NDVI, land surface temperature) are explicitly categorized as **"Operational / Environmental Risk & Surface Alteration Proxies"**, rather than falsely claiming they directly detect underground manganese mineralization. Subsurface prospectivity is strictly driven by lithology, structural lineament density, faults, distance to known deposits, and hydrothermal alteration indices.

> [!NOTE]
> **Reserve Classification Transparency**: AI outputs are designated as **"AI Prospectivity Estimates"** and **"Exploration Targets"**. They do *not* constitute statutory certified mineral reserves under UNFC (United Nations Framework Classification) or JORC codes, which require physical core assay validation.

---

## 3. Platform Architecture

```
moil-manganese-twin/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application (12 REST endpoints, CORS, static server)
│   │   └── static/
│   │       └── index.html       # Industrial dark command centre single-page application
│   └── run_server.py            # Local server launcher
├── ml/
│   ├── ml_core.py               # Offline high-performance Pure ML algorithms (RF, XGBoost, GBDT, LogReg, IDW)
│   ├── data_generator.py        # Seed: 26009 (5-yr daily production, 125 drillholes, 400 raster cells)
│   ├── prospectivity_engine.py  # Engine A: Multi-model training, ROC-AUC metrics, zone prediction
│   ├── production_engine.py     # Engine B: 30-day forecaster, SARIMA vs RF vs XGBoost, R² metrics
│   ├── root_cause_engine.py     # Explainable AI (XAI): TreeSHAP shortfall attribution breakdown
│   ├── subsurface_interpolator.py # 3D geological block model & Monte Carlo reserve tonnage
│   └── optimizer_engine.py      # Operations Research (OR) constraint optimizer (+5,300 T recovery)
├── data/
│   ├── geological/              # Geological zones JSON & 400-cell prospectivity raster CSV
│   ├── drilling/                # 125 synthetic drillholes with stratigraphic intercepts & Mn grades
│   ├── production/              # 1,826 daily records (5 years) with weather, fleet & shortfall labels
│   └── equipment/               # Fleet digital twin status (EX-01..02, HT-01..06, DR-01..02, CR-01)
├── models/                      # Serialized trained model weights & evaluation JSON files
├── tests/
│   └── test_api_endpoints.py    # Automated test suite (8 test cases covering all scenarios)
├── start_demo.bat               # 1-Click Windows launch script
└── README.md                    # Technical documentation & presentation guide
```

---

## 4. Key AI & Digital Twin Engines

### Engine A — Manganese Prospectivity & Reserve Intelligence
- **Geological Features**: Structural fault density, lineament intersection density, host lithology favorability (Mansar Schist / Gondite contact), hydrothermal spectral alteration index, proximity to known MOIL deposits, DEM elevation, topographic slope.
- **Trained Models**: Random Forest (ROC-AUC: 0.994), XGBoost (ROC-AUC: 0.977), Gradient Boosting (ROC-AUC: 0.966), Logistic Regression (ROC-AUC: 0.998).
- **Spatial Grid**: 400 raster cells categorized as VERY HIGH (>80%), HIGH (60–80%), MEDIUM (40–60%), LOW (<40%).
- **Interactive Drill Simulator**: Click any zone to simulate exploratory core drilling (e.g. Zone A yields 15m ore intercept at 34.2% Mn grade).
- **3D Block Model**: Interactive 3D voxel orebody viewer with depth slider (0 to 90m) and Monte Carlo reserve estimation ($2.84\text{ MT} \pm 0.52\text{ MT}$).

### Engine B — Production Shortfall Forecaster & Root Cause
- **Time-Series Forecaster**: 30-day forward forecast with 90% confidence bands comparing SARIMA baseline against Random Forest and XGBoost Regressor (Best Model, $R^2 = 0.581$, $\text{MAPE} = 7.8\%$).
- **Explainable AI (TreeSHAP)**: Isolates the root causes of production shortfalls:
  - Equipment Downtime: **38%** (-1,994 T)
  - Monsoon Weather & Haul Slip: **24%** (-1,259 T)
  - Blasting Schedule Delays: **19%** (-997 T)
  - Crusher Throughput Bottlenecks: **12%** (-630 T)
  - Haulage Detour Inefficiencies: **7%** (-368 T)

### Operations Research (OR) Optimization Layer
Calculates prioritized, constraint-compliant operational levers that recover up to **+5,300 Tonnes**:
1. **Redeploy Haul Truck HT-04 to Zone B High-Grade Loop**: $+2,100\text{ T}$ (Shortens round trip by 1.2 km).
2. **Reschedule Blast B-17 to Tomorrow 14:00 Window**: $+1,400\text{ T}$ (Prevents shovel starvation at Bench 4).
3. **Expedite EX-02 Hydraulic Seal PM (12h Early Release)**: $+1,000\text{ T}$ (Restores primary shovel digging capacity).
4. **Extend Primary Crusher CR-01 Operating Window by 1.5h**: $+800\text{ T}$ (Absorbs truck delivery surges).

---

## 5. Live Local Execution

The server is running locally on your machine at:
- **Local Application URL**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

To launch anytime from terminal or file explorer:
```powershell
# Option 1: Double click or run batch file
.\start_demo.bat

# Option 2: Run via python directly
python run_server.py
```

To run the automated test suite:
```powershell
python -m pytest tests/test_api_endpoints.py -v
```

---

## 6. The 9-Step SIH Hackathon Demo Narrative (Judge Mode)
Click the golden **"⭐ START SIH DEMO / JUDGE MODE"** button in the header to guide judges through the 3-minute pitch:
1. **Step 1**: AI scans geological lineaments and multi-spectral satellite layers across 400 raster cells.
2. **Step 2**: Two high-prospectivity exploration horizons identified (Zone A: 91.8%, Zone D: 89.4%).
3. **Step 3**: 30-day production forecast evaluated against the monthly 52,000 T MOIL target.
4. **Step 4**: Monsoon cloudburst (182 mm) + EX-02 shovel downtime triggers an **82.9% Shortfall Alert** (-5,248 T gap).
5. **Step 5**: Explainable AI identifies dominant constraints (38% Shovel downtime, 24% Weather haul slip, 19% Blast delay).
6. **Step 6**: Digital twin tests dispatch interventions across the 11-node mining fleet.
7. **Step 7**: AI recommends 4 quantified interventions (+5,300 tonnes potential recovery).
8. **Step 8**: Digital twin applies interventions, eliminating the shortfall and lowering risk from 82.9% to 14.5%.
9. **Step 9**: Chief Mine Planner authorizes the immutable decision log into the statutory audit trail.
