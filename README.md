# Multi-Agent Traffic Light Control System

A small-scale multi-agent system (MAS) that replaces fixed-time traffic lights with learning agents using tabular Q-learning to optimize traffic flow in an urban road network simulated in SUMO.

## Project Overview

This project implements a distributed traffic control system where each signalized intersection is controlled by an autonomous SPADE agent that learns optimal signal timing through reinforcement learning. The agents aim to minimize average queue length and delay across the network.

### Key Features

- **Multi-Agent Architecture**: Each traffic light is controlled by an independent SPADE agent
- **Tabular Q-Learning**: Agents learn optimal policies offline through interaction with the environment
- **SUMO Integration**: Realistic traffic simulation using the SUMO traffic simulator
- **Real-time Coordination**: Agents communicate statistics to a coordinator for monitoring
- **Scalable Design**: Easy to extend to larger networks or different RL algorithms

### Network Layout

The simulated network contains:
- **4 Traffic Lights** (2 T-junctions + 2 X-junctions)
- **2 Signal Phases** per intersection:
  - Phase 1: North-South green, East-West red
  - Phase 2: East-West green, North-South red

## Installation

### Prerequisites

1. **Python 3.8+** with conda or venv
2. **SUMO 1.23+** installed and `SUMO_HOME` environment variable set
3. **Windows/Linux/macOS** (tested on Windows 10)

No Docker or external XMPP server needed - SPADE starts its own built-in XMPP server automatically!

### Setup Instructions

1. **Clone the repository** (or extract the project files)

2. **Create and activate conda environment**:
   ```bash
   conda create -n traffic-mas python=3.9
   conda activate traffic-mas
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify SUMO installation**:
   ```bash
   echo %SUMO_HOME%  # Windows
   echo $SUMO_HOME   # Linux/macOS
   ```

## Usage

### Step 1: Train the Q-Learning Agents

Train the Q-tables offline using tabular Q-learning:

```bash
cd train
python train_qlearn.py
```

This will:
- Run 400 training episodes (configurable)
- Save Q-tables to `models/q_tables.pkl`
- Display training progress and average rewards
- Takes approximately 15-30 minutes depending on your system

### Step 2: Run Baseline Simulation (Optional but Recommended)

Run a baseline simulation with fixed-time traffic lights for comparison:

```bash
cd ..
python run_baseline_sim.py --duration 600  # 10-minute test
```

This will:
- Run SUMO with default fixed-time traffic light programs
- Save statistics to `output/baseline/`
- Show progress every 10 seconds

### Step 3: Run the Multi-Agent Simulation

Run the RL-controlled simulation with SPADE agents:

```bash
python run_sim.py --duration 600  # 10-minute test
```

This will:
- Start SUMO GUI with the traffic network
- Launch SPADE agents with built-in XMPP server (starts automatically!)
- Launch a coordinator agent for statistics collection
- Save statistics to `output/rl/`
- Display real-time statistics every 30 seconds

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

That's it! No cleanup needed - the built-in XMPP server stops automatically when the simulation ends.

### Command-Line Options

To run without GUI (faster):
```python
# In run_sim.py, change:
sim = TrafficSimulation(use_gui=False)
```

To change simulation duration:
```python
# In run_sim.py, modify:
await sim.run_simulation(duration=3600)  # 1 hour
```

## Project Structure

```
traffic-mas/
├── agents/
│   ├── __init__.py
│   ├── intersection.py    # SPADE agent for traffic lights
│   └── coordinator.py     # Statistics collection agent
├── models/
│   └── q_tables.pkl      # Trained Q-tables (generated)
├── nets/
│   ├── test.net.xml      # SUMO network definition
│   ├── routes.rou.xml    # Traffic demand routes
│   └── trips.trips.xml   # Trip definitions
├── train/
│   ├── train_qlearn.py   # Q-learning training script
│   └── train_qlearn_v2.py # Alternative training (version-robust)
├── output/               # Simulation results (generated)
│   ├── baseline/         # Baseline simulation outputs
│   └── rl/              # RL simulation outputs
├── run_sim.py          # RL simulation launcher (with built-in XMPP server!)
├── run_baseline_sim.py # Baseline simulation launcher
├── compare_results.py  # Compare baseline vs RL results
├── requirements.txt    # Python dependencies (simplified!)
└── README.md           # This file
```

## How It Works

### Training Phase

1. **Environment Setup**: Uses `sumo-rl` to wrap SUMO as a Gym environment
2. **State Space**: Discretized queue lengths for each approach (5 bins: 0, 1-3, 4-6, 7-10, >10)
3. **Action Space**: Binary - keep current phase or switch to next phase
4. **Reward Function**: Negative sum of queue lengths at the intersection
5. **Q-Learning**: Updates Q-values using Bellman equation with α=0.1, γ=0.95

### Runtime Phase

1. **Agent Initialization**: Each traffic light gets a SPADE agent with its trained Q-table
2. **Decision Making**: Every 5 seconds, agents observe queue lengths and decide actions
3. **Coordination**: Agents report statistics to coordinator every 10 seconds
4. **Visualization**: SUMO GUI shows traffic flow, coordinator prints aggregate statistics

## Monitoring and Results

During simulation, you'll see:

1. **SUMO GUI**: Visual representation of traffic flow and signal states
2. **Console Output**: 
   - Agent status messages
   - Periodic traffic statistics
   - Queue lengths and waiting times per intersection

3. **Output Files**:
   - `output/tripinfo.xml`: Detailed trip statistics for analysis

## Extending the Project

### Add Deep Q-Learning (DQN)

Replace tabular Q-learning with neural networks:
1. Modify `train_qlearn.py` to use `stable-baselines3` DQN
2. Update `IntersectionAgent` to load and use neural network models

### Scale to Larger Networks

1. Generate larger SUMO networks using `netgenerate`
2. Adjust state discretization for more complex intersections
3. Consider hierarchical coordination between agents

### Add More Sophisticated Rewards

- Include waiting time in reward function
- Add penalties for frequent phase changes
- Consider global vs. local reward optimization

## Troubleshooting

### Common Issues

1. **"SUMO_HOME not set"**: Set the environment variable to your SUMO installation
2. **"No module named 'traci'"**: Ensure SUMO Python tools are in your PATH
3. **"Q-tables not found"**: Run training script before simulation
4. **Docker not running**: Start Docker Desktop (Windows/macOS) or Docker daemon (Linux)
5. **Port already in use**: Stop any service using ports 5222, 5280, or 5269
6. **SPADE connection errors**: Make sure ejabberd is running (`docker ps` should show the container)

### Performance Tips

- Use `use_gui=False` for faster training
- Reduce `delay` in SUMO for quicker visualization
- Adjust `decision_interval` in agents for different control frequencies

## References

- SUMO Documentation: https://sumo.dlr.de/docs/
- SPADE Framework: https://spade-mas.readthedocs.io/
- SUMO-RL: https://github.com/LucasAlegre/sumo-rl

## License

This project is created for educational purposes as part of an Intelligent Systems & Machine Learning (ISML) assignment. 