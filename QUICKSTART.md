# Quick Start Guide - Traffic MAS

## Prerequisites Check
```bash
python test_setup.py
```

## Step-by-Step Instructions

### 1. Train the Agents (First Time Only)
```bash
cd train
python train_qlearn_v2.py  # Use this if you get version errors
# or
python train_qlearn.py
cd ..
```
This takes ~15-30 minutes. You only need to do this once.

If you get errors, first test your environment:
```bash
cd train
python test_sumo_rl.py
```

### 2. Run Baseline Simulation (For Comparison)
```bash
python run_baseline_sim.py --duration 600  # 10 minutes for quick test
# or
python run_baseline_sim.py  # Full 30 minutes
```
This runs with fixed-time traffic lights (no RL).

### 3. Run RL-Controlled Simulation
```bash
python run_sim.py --duration 600  # 10 minutes for quick test
# or
python run_sim.py  # Full 30 minutes
```
The SUMO GUI will open and agents will control traffic lights using SPADE's built-in XMPP server.

### 4. Compare Results
```bash
python compare_results.py
```
This shows the improvements achieved by RL control.

## Troubleshooting

### SUMO Issues
- Make sure `SUMO_HOME` environment variable is set
- Test with: `echo $SUMO_HOME` (Linux/macOS) or `echo %SUMO_HOME%` (Windows)

### Python Dependencies
If you get import errors:
```bash
pip install -r requirements.txt
```

### Reset Everything
```bash
rm -rf models/*.pkl     # Remove trained models to retrain
rm -rf output/          # Clear previous results
```

## What's New?
- **No Docker needed!** - Uses SPADE's built-in XMPP server  
- **No external setup** - Everything runs locally
- **Automatic startup** - XMPP server starts automatically
- **Simpler deployment** - Just Python and SUMO required