from datetime import datetime
import os
import sys
import asyncio
import time
from threading import Thread
import traci
import spade
from agents.intersection import IntersectionAgent
from agents.coordinator import CoordinatorAgent

# SUMO configuration
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")


class TrafficSimulation:    
    def __init__(self, use_gui=True, duration=1800):
        self.use_gui = use_gui
        self.duration = duration
        self.agents = []
        self.coordinator = None
        self.sumo_thread = None
        self.running = False
        self.timestamp = None
        
    def start_sumo(self):
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        
        os.makedirs("output/rl", exist_ok=True)        
        from datetime import datetime
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        sumo_cmd = [
            sumo_binary,
            "-n", "nets/test.net.xml",
            "-r", "nets/routes.rou.xml",
            ("--start" if self.use_gui else "--no-step-log"),
            "--quit-on-end",
            "--delay", "50" if self.use_gui else "0",
            "--time-to-teleport", "-1",
            "--no-warnings",
            "--duration-log.statistics",
            "--tripinfo-output", f"output/rl/tripinfo_{self.timestamp}.xml",
            "--summary-output", f"output/rl/summary_{self.timestamp}.xml",
            "--statistic-output", f"output/rl/statistics_{self.timestamp}.xml",
            "--queue-output", f"output/rl/queue_{self.timestamp}.xml",
            "--summary-output.period", "60",
            "--queue-output.period", "60",
            "--end", str(self.duration)
        ]
        
        traci.start(sumo_cmd)
        print("SUMO started successfully with RL agents")
        print(f"Output files will be saved to: output/rl/")
        
        return self.timestamp
        
    def create_agents(self):
        tls_ids = traci.trafficlight.getIDList()
        print(f"Found traffic lights: {tls_ids}")
        
        self.coordinator = CoordinatorAgent(
            jid="coordinator@localhost", 
            password="password"
        )
        
        for i, tls_id in enumerate(tls_ids):
            agent_name = f"tls_{tls_id}".lower()
            agent_jid = f"{agent_name}@localhost"
            agent = IntersectionAgent(
                jid=agent_jid,
                password="password",
                tls_id=tls_id,
                q_table_path=os.path.join(os.path.dirname(__file__), "models", "q_tables.pkl")
            )
            self.agents.append(agent)
        
        return tls_ids
    
    async def start_agents(self):
        print("Starting SPADE agents...")
        
        await self.coordinator.start()
        print("Coordinator agent started")
        
        for agent in self.agents:
            await agent.start()
            print(f"Started agent for {agent.tls_id}")
        
        print("All agents started successfully!")
        await asyncio.sleep(2)
    
    def run_simulation_step(self):
        traci.simulationStep()
        
    async def run_simulation(self):
        self.running = True
        step = 0
        
        print(f"\nStarting RL-controlled simulation for {self.duration} seconds...")
        print("Press Ctrl+C to stop the simulation\n")
        
        try:
            while self.running and traci.simulation.getTime() < self.duration:
                try:
                    self.run_simulation_step()
                    await asyncio.sleep(0.1)
                    
                    step += 1
                    
                    if step % 100 == 0:
                        sim_time = traci.simulation.getTime()
                        vehicle_count = traci.vehicle.getIDCount()
                        waiting_count = 0
                        total_waiting_time = 0
                        
                        for veh_id in traci.vehicle.getIDList():
                            speed = traci.vehicle.getSpeed(veh_id)
                            waiting_time = traci.vehicle.getWaitingTime(veh_id)
                            if speed < 0.1:
                                waiting_count += 1
                            total_waiting_time += waiting_time
                        
                        avg_waiting = total_waiting_time / vehicle_count if vehicle_count > 0 else 0
                        progress = (sim_time / self.duration) * 100
                        
                        print(f"Time: {sim_time:.0f}s ({progress:.1f}%) | "
                              f"Vehicles: {vehicle_count} | "
                              f"Waiting: {waiting_count} | "
                              f"Avg Wait: {avg_waiting:.1f}s")
                        
                except traci.exceptions.FatalTraCIError:
                    print("SUMO simulation ended")
                    break
                    
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        
        self.running = False
    
    async def stop_agents(self):
        print("\nStopping agents...")
        
        for agent in self.agents:
            await agent.stop()
        
        if self.coordinator:
            await self.coordinator.stop()
        
        print("All agents stopped")
    
    def cleanup(self):
        try:
            traci.close()
        except:
            pass
        
        print("Simulation cleanup complete")


async def main():
    if not os.path.exists("models/q_tables.pkl"):
        print("ERROR: Q-tables not found! Please run train/train_qlearn.py first.")
        return
    
    print("Traffic MAS starting with SPADE built-in XMPP server...")
    
    os.makedirs("output", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"output/sim_{timestamp}.log"
    
    # Create a logger that writes to both console and file
    import sys
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    print(f"Logging to: {log_file}")
    
    import argparse
    parser = argparse.ArgumentParser(description="Run RL-controlled traffic simulation")
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI")
    parser.add_argument("--duration", type=int, default=1800, 
                        help="Simulation duration in seconds (default: 1800)")
    args = parser.parse_args()
    
    sim = TrafficSimulation(use_gui=args.gui, duration=args.duration)
    
    try:
        timestamp = sim.start_sumo()        
        tls_ids = sim.create_agents()        
        await sim.start_agents()        
        await sim.run_simulation()
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        
    finally:
        await sim.stop_agents()        
        sim.cleanup()
        
        print("\nRL-controlled simulation completed!")        
        print(f"\nOutput files saved with timestamp: {timestamp}")
        print("Files:")
        print(f"  - output/rl/tripinfo_{timestamp}.xml")
        print(f"  - output/rl/summary_{timestamp}.xml")
        print(f"  - output/rl/statistics_{timestamp}.xml")
        print(f"  - output/rl/queue_{timestamp}.xml")

if __name__ == "__main__":
    # Launches SPADE native XMPP broker
    spade.run(main(), True)