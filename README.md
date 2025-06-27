# Multi-Agent Traffic Light Control System

A small-scale multi-agent system (MAS) that replaces fixed-time traffic lights with learning agents using tabular Q-learning (reinforced learning) to optimize traffic flow in a road network simulated in SUMO.

## Installation

### Setup Instructions

1. **Install Anaconda**

Head to `https://www.anaconda.com/download`.

Download and install conda.

Go to start menu, search and launch anaconda prompt.

2. **Install SUMO**

Head to `https://sumo.dlr.de/docs/Downloads.php`.

Download and run the 64-bit installer.

Test that it works either by running sumo-gui from start menu or running `sumo-gui` in terminal.

2. **Clone the repository** (or extract the project files)

3. **Create and activate conda environment**:
   ```bash
   conda create -n traffic_light python=3.11
   conda activate traffic_light
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify SUMO installation**:
   ```bash
   echo %SUMO_HOME%
   ```

## Usage

### OPTIONAL STEP: Create your own road network and routes

Launch netedit.

Create your own road network.

Save it as test.net.xml, overwriting the one provided by the repo.

In the terminal, run

```bash
python "%SUMO_HOME%\tools\randomTrips.py" ^
-n test.net.xml -r routes.rou.xml ^
-b 0 -e 1800 -p 3 ^
--fringe-factor 10 --seed 42
```

This will generate a routes.rou.xml file and a trips.trips.xml (remains unused) file

Test configuration correctness by running

```bash
sumo-gui -n test.net.xml -r routes.rou.xml
```

This should start SUMO's own fixed phase duration simulation of the network.

### Step 1: Train the Q-Learning Agents (optional)

Train the Q-tables using tabular Q-learning:

```bash
cd train
python train_qlearn.py
```

This will:
- Run 400 training episodes (configurable)
- Save Q-tables to `models/q_tables.pkl`
- Display training progress and average rewards
- Takes around an hour

### Step 2: Run Baseline Simulation (Optional but Recommended)

Run a baseline simulation with fixed-time traffic lights for comparison:

```bash
cd ..
python run_baseline_sim.py --duration 1800
```

**OR**

```bash
cd ..
python run_baseline_sim.py --duration 1800 --gui
```

This will:
- Run SUMO with default fixed-time traffic light programs
- Save statistics to `output/baseline/`

### Step 3: Run the Multi-Agent Simulation

Run the RL-controlled simulation with SPADE agents:

```bash
python run_sim.py --duration 1800
```

**OR**

```bash
python run_sim.py --duration 1800 --gui
```

This will:
- Start SUMO GUI with the traffic network
- Launch SPADE native XMPP broker
- Launch SPADE agents
- Launch a coordinator agent for statistics collection
- Save statistics to `output/rl/`

### Step 4: Compare Results

Compare the baseline and RL simulations:

```bash
python compare_results.py
```

This will show:
- Average waiting time reduction
- Travel time improvements
- Speed increases
- Total time saved