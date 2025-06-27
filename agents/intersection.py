import asyncio
import os
import pickle
import numpy as np
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message
import traci

class IntersectionAgent(Agent):    
    def __init__(self, jid, password, tls_id, q_table_path):
        super().__init__(jid, password)
        self.tls_id = tls_id
        self.q_table_path = q_table_path
        self.q_table = None
        self.decision_interval = 5  # seconds between decisions
        self.current_phase = 0
        self.time_in_phase = 0
        self.min_green = 5
        self.yellow_time = 3
        
    async def setup(self):
        print(f"IntersectionAgent {self.tls_id} starting...")
        
        self.load_q_table()        
        await asyncio.sleep(1)

        # Sumo moment. Test if the agents can modify light phase        
        try:
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            current_program = traci.trafficlight.getProgram(self.tls_id)
            controlled_lanes = len(set(traci.trafficlight.getControlledLanes(self.tls_id)))
            
            print(f"  {self.tls_id} status: phase={current_phase}, program={current_program}, controls {controlled_lanes} lanes")
            
            phases = traci.trafficlight.getAllProgramLogics(self.tls_id)[0].phases
            if len(phases) > 1:
                old_phase = current_phase
                test_phase = (current_phase + 2) % len(phases)
                
                print(f"  {self.tls_id}: Testing control capability - changing phase {old_phase} -> {test_phase}")
                traci.trafficlight.setPhase(self.tls_id, test_phase)
                
                await asyncio.sleep(0.5)
                
                new_phase = traci.trafficlight.getPhase(self.tls_id)
                if new_phase == test_phase:
                    print(f"  {self.tls_id}: Control test SUCCESSFUL ✓")
                else:
                    print(f"  {self.tls_id}: Control test FAILED ✗ (phase {new_phase} != requested {test_phase})")
                
                traci.trafficlight.setPhase(self.tls_id, old_phase)
            
        except Exception as e:
            print(f"  Error verifying control of {self.tls_id}: {e}")
        
        traffic_behaviour = TrafficControlBehaviour(decision_interval=self.decision_interval)
        self.add_behaviour(traffic_behaviour)
        
        stats_behaviour = StatsReportingBehaviour(period=10)
        self.add_behaviour(stats_behaviour)
        
        print(f"IntersectionAgent {self.tls_id} initialized successfully")
        
    def load_q_table(self):
        print(f"Loading Q-table from {self.q_table_path}")
        try:
            with open(self.q_table_path, 'rb') as f:
                all_q_tables = pickle.load(f)
                if self.tls_id in all_q_tables:
                    self.q_table = all_q_tables[self.tls_id]
                    print(f"Loaded Q-table for {self.tls_id} with {len(self.q_table)} states")
                else:
                    print(f"Warning: No Q-table found for {self.tls_id}, using random policy")
                    self.q_table = {}
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            self.q_table = {}
    
    def discretize_queue(self, queue_length):
        if queue_length == 0:
            return 0
        elif queue_length <= 3:
            return 1
        elif queue_length <= 6:
            return 2
        elif queue_length <= 10:
            return 3
        else:
            return 4
    
    def get_state(self):
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
            
            queue_lengths = []
            unique_lanes = sorted(set(controlled_lanes))  # Remove duplicates
            
            for lane in unique_lanes:
                halting_vehicles = traci.lane.getLastStepHaltingNumber(lane)
                queue_lengths.append(self.discretize_queue(halting_vehicles))
            
            # Add padding to achieve 4 dimensions
            while len(queue_lengths) < 4:
                queue_lengths.append(0)
            
            queue_lengths = queue_lengths[:4]
            
            return tuple(queue_lengths)
        except Exception as e:
            print(f"Error getting state: {e}")
            return (0, 0, 0, 0)
    
    def get_action(self, state):
        if state in self.q_table:
            # Greedy action selection
            q_values = self.q_table[state]
            if isinstance(q_values, dict):
                # Handle case where q_values might be stored as dict
                action = max(q_values.keys(), key=lambda a: q_values[a])
            else:
                # If Q-values are very close (difference < 0.1), prefer switching
                if len(q_values) >= 2 and abs(q_values[0] - q_values[1]) < 0.1:
                    action = 1  # Prefer switching when uncertain
                else:
                    action = np.argmax(q_values)
            return action
        else:
            # Handle specific missing states
            if state == (0, 0, 0, 0):
                # All lanes empty, limit switch
                import random
                action = 0 if random.random() < 0.8 else 1
                if hasattr(self, '_empty_state_log_count'):
                    self._empty_state_log_count += 1
                    if self._empty_state_log_count % 10 == 1:
                        print(f"Empty state for {self.tls_id}: {state}, using conservative policy (action {action})")
                else:
                    self._empty_state_log_count = 1
                    print(f"Empty state for {self.tls_id}: {state}, using conservative policy (action {action})")
                return action
            else:
                # Try to match unknown state
                similar_action = self._find_similar_state_action(state)
                if similar_action is not None:
                    return similar_action
                
                print(f"Unknown state for {self.tls_id}: {state}, defaulting to action 1 (switch)")
                return 1
    
    def _find_similar_state_action(self, target_state):
        if not self.q_table:
            return None
            
        best_similarity = -1
        best_action = None
        
        for known_state, q_values in self.q_table.items():
            similarity = sum(1 for i, j in zip(target_state, known_state) if i == j)
            if similarity > best_similarity:
                best_similarity = similarity
                if isinstance(q_values, dict):
                    best_action = max(q_values.keys(), key=lambda a: q_values[a])
                else:
                    best_action = np.argmax(q_values)
        
        if best_similarity >= 2:
            return best_action
        return None


class TrafficControlBehaviour(CyclicBehaviour):    
    def __init__(self, decision_interval):
        super().__init__()
        self.last_decision_time = 0
        self.decision_interval=decision_interval
        self.last_phase = None
        self.phase_start_time = None
        
    async def run(self):
        try:
            current_time = traci.simulation.getTime()
            if current_time - self.last_decision_time < self.decision_interval:
                return
            self.last_decision_time = current_time

            state = self.agent.get_state()
            current_phase = traci.trafficlight.getPhase(self.agent.tls_id)
            
            if self.last_phase != current_phase:
                self.phase_start_time = traci.simulation.getTime()
                self.last_phase = current_phase
                self.agent.time_in_phase = 0
            else:
                self.agent.time_in_phase = traci.simulation.getTime() - self.phase_start_time
            
            current_state = traci.trafficlight.getRedYellowGreenState(self.agent.tls_id)
            is_yellow = 'y' in current_state.lower()
            
            if is_yellow:
                return
            
            if self.agent.time_in_phase >= self.agent.min_green:
                action = self.agent.get_action(state)
                
                if action == 1:
                    phases = traci.trafficlight.getAllProgramLogics(self.agent.tls_id)[0].phases
                    num_phases = len(phases)
                    
                    next_phase = (current_phase + 1) % num_phases
                    attempts = 0
                    while 'y' in phases[next_phase].state.lower() and attempts < num_phases:
                        next_phase = (next_phase + 1) % num_phases
                        attempts += 1
                    
                    if attempts < num_phases:
                        traci.trafficlight.setPhase(self.agent.tls_id, next_phase)
                        self.phase_start_time = traci.simulation.getTime()
                        self.agent.time_in_phase = 0
                        self.agent.current_phase = next_phase
                        self.last_phase = next_phase
                        
                        print(f"### PHASE CHANGE ### {self.agent.tls_id}: Switching to phase {next_phase} (state: {state}), sim time: {traci.simulation.getTime():.1f}s")
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error in traffic control: {e}")
    
    async def on_start(self):
        print(f"Traffic control behaviour started for {self.agent.tls_id}")
    
    async def on_end(self):
        print(f"Traffic control behaviour ended for {self.agent.tls_id}")


class StatsReportingBehaviour(PeriodicBehaviour):
    def __init__(self, period):
        super().__init__(period=period)

    async def run(self):
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.agent.tls_id)
            unique_lanes = list(set(controlled_lanes))
            
            total_waiting = 0
            total_queue = 0
            
            for lane in unique_lanes:
                total_waiting += max(0, traci.lane.getWaitingTime(lane))
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
            
            msg = Message(to="coordinator@localhost")  
            msg.set_metadata("performative", "inform")
            msg.body = f"{self.agent.tls_id}|queue:{total_queue}|waiting:{total_waiting:.2f}"
            
            await self.send(msg)
            
        except Exception as e:
            print(f"Error reporting stats: {e}")
    
    async def on_start(self):
        print(f"Stats reporting behaviour started for {self.agent.tls_id}")
    
    async def on_end(self):
        print(f"Stats reporting behaviour ended for {self.agent.tls_id}") 