"""
Baseline simulation with fixed-time traffic lights
Used for comparison with RL-controlled traffic lights
"""

import os
import sys
import time
import traci
from datetime import datetime

# SUMO configuration
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")


class BaselineSimulation:
    """Run SUMO simulation with default fixed-time traffic lights"""
    
    def __init__(self, use_gui=False, duration=1800):
        self.use_gui = use_gui
        self.duration = duration  # Simulation duration in seconds
        self.start_time = None
        
    def start_sumo(self):
        """Start SUMO simulation with statistics output"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        
        # Create output directory
        os.makedirs("output/baseline", exist_ok=True)
        
        # Get current timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        sumo_cmd = [
            sumo_binary,
            "-n", "nets/test.net.xml",
            "-r", "nets/routes.rou.xml",
            ("--start" if self.use_gui else "--no-step-log"),
            "--quit-on-end",
            "--delay", "50" if self.use_gui else "0",
            "--time-to-teleport", "-1",  # Disable teleporting
            "--no-warnings",
            "--duration-log.statistics", "true",
            "--tripinfo-output", f"output/baseline/tripinfo_{timestamp}.xml",
            "--summary-output", f"output/baseline/summary_{timestamp}.xml",
            "--statistic-output", f"output/baseline/statistics_{timestamp}.xml",
            "--queue-output", f"output/baseline/queue_{timestamp}.xml",
            "--summary-output.period", "60",  # Output summary every 60 seconds
            "--queue-output.period", "60",  # Queue output every 60 seconds
            "--end", str(self.duration)
        ]
        
        traci.start(sumo_cmd)
        print("Baseline SUMO simulation started")
        print(f"Output files will be saved to: output/baseline/")
        
        return timestamp
        
    def run_simulation(self):
        """Run the baseline simulation"""
        self.start_time = time.time()
        step = 0
        
        print(f"\nRunning baseline simulation for {self.duration} seconds...")
        print("Using default fixed-time traffic light programs")
        print("Press Ctrl+C to stop the simulation\n")
        
        try:
            while traci.simulation.getTime() < self.duration:
                traci.simulationStep()
                step += 1
                
                # Print progress every 100 steps (10 seconds)
                if step % 100 == 0:
                    sim_time = traci.simulation.getTime()
                    vehicle_count = traci.vehicle.getIDCount()
                    waiting_count = 0
                    total_waiting_time = 0
                    
                    # Calculate waiting vehicles
                    for veh_id in traci.vehicle.getIDList():
                        speed = traci.vehicle.getSpeed(veh_id)
                        waiting_time = traci.vehicle.getWaitingTime(veh_id)
                        if speed < 0.1:  # Vehicle is stopped
                            waiting_count += 1
                        total_waiting_time += waiting_time
                    
                    avg_waiting = total_waiting_time / vehicle_count if vehicle_count > 0 else 0
                    progress = (sim_time / self.duration) * 100
                    
                    print(f"Time: {sim_time:.0f}s ({progress:.1f}%) | "
                          f"Vehicles: {vehicle_count} | "
                          f"Waiting: {waiting_count} | "
                          f"Avg Wait: {avg_waiting:.1f}s")
                          
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except traci.exceptions.FatalTraCIError:
            print("Simulation ended")
        
        elapsed = time.time() - self.start_time
        print(f"\nBaseline simulation completed in {elapsed:.1f} seconds")
        
    def print_traffic_light_info(self):
        """Print information about traffic lights and their programs"""
        tls_ids = traci.trafficlight.getIDList()
        print(f"\nTraffic lights in network: {tls_ids}")
        
        for tls_id in tls_ids:
            program = traci.trafficlight.getProgram(tls_id)
            phase_idx = traci.trafficlight.getPhase(tls_id)
            print(f"\n{tls_id}:")
            print(f"  Program: {program}")
            print(f"  Current phase: {phase_idx}")
            
            # Get phase definitions
            logic = traci.trafficlight.getAllProgramLogics(tls_id)
            if logic:
                phases = logic[0].phases
                print(f"  Total phases: {len(phases)}")
                for i, phase in enumerate(phases[:4]):  # Show first 4 phases
                    print(f"    Phase {i}: duration={phase.duration}s, state={phase.state}")
    
    def cleanup(self):
        """Clean up simulation"""
        try:
            traci.close()
        except:
            pass
        
        print("Baseline simulation cleanup complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline SUMO simulation")
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI")
    parser.add_argument("--duration", type=int, default=1800, 
                        help="Simulation duration in seconds (default: 1800)")
    args = parser.parse_args()
    
    # Create and run simulation
    sim = BaselineSimulation(use_gui=args.gui, duration=args.duration)
    
    try:
        # Start SUMO
        timestamp = sim.start_sumo()
        
        # Print traffic light information
        sim.print_traffic_light_info()
        
        # Run simulation
        sim.run_simulation()
        
        print(f"\nOutput files saved with timestamp: {timestamp}")
        print("Files:")
        print(f"  - output/baseline/tripinfo_{timestamp}.xml")
        print(f"  - output/baseline/summary_{timestamp}.xml")
        print(f"  - output/baseline/statistics_{timestamp}.xml")
        print(f"  - output/baseline/queue_{timestamp}.xml")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        
    finally:
        # Cleanup
        sim.cleanup()


if __name__ == "__main__":
    main() 