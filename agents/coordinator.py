import asyncio
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.template import Template
from datetime import datetime
import numpy as np


class CoordinatorAgent(Agent):
    """SPADE agent that coordinates and monitors all traffic light agents"""
    
    def __init__(self, jid, password):
        super().__init__(jid, password)
        self.statistics = {}  # Store stats for each intersection
        self.start_time = None
        
    async def setup(self):
        """Initialize coordinator agent"""
        print(f"CoordinatorAgent starting...")
        self.start_time = datetime.now()
        
        # Wait a bit to ensure proper connection
        await asyncio.sleep(1)
        
        # Add message receiving behaviour
        receive_behaviour = ReceiveStatsBehaviour()
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(receive_behaviour, template)
        
        # Add periodic reporting behaviour
        report_behaviour = ReportStatsBehaviour(period=30)  # Report every 30 seconds
        self.add_behaviour(report_behaviour)
        
        print("CoordinatorAgent initialized successfully")


class ReceiveStatsBehaviour(CyclicBehaviour):
    """Behaviour that receives statistics from intersection agents"""
    
    def __init__(self):
        super().__init__()
    
    async def run(self):
        """Receive and process statistics messages"""
        msg = await self.receive(timeout=1)  # 1 second timeout
        
        if msg:
            try:
                # Parse message body: "TLS_ID|queue:X|waiting:Y"
                parts = msg.body.split('|')
                tls_id = parts[0]
                
                stats = {}
                for part in parts[1:]:
                    key, value = part.split(':')
                    stats[key] = float(value)
                
                # Update statistics
                if tls_id not in self.agent.statistics:
                    self.agent.statistics[tls_id] = {
                        'queue_history': [],
                        'waiting_history': [],
                        'last_update': datetime.now()
                    }
                
                self.agent.statistics[tls_id]['queue_history'].append(stats['queue'])
                self.agent.statistics[tls_id]['waiting_history'].append(stats['waiting'])
                self.agent.statistics[tls_id]['last_update'] = datetime.now()
                
                # Keep only last 100 values to avoid memory issues
                if len(self.agent.statistics[tls_id]['queue_history']) > 100:
                    self.agent.statistics[tls_id]['queue_history'].pop(0)
                    self.agent.statistics[tls_id]['waiting_history'].pop(0)
                
            except Exception as e:
                print(f"Error processing message: {e}")
    
    async def on_start(self):
        """Called when behaviour starts"""
        print("Coordinator started receiving statistics")


class ReportStatsBehaviour(PeriodicBehaviour):
    """Behaviour that periodically reports aggregated statistics"""

    def __init__(self, period):
        super().__init__(period=period)
    
    async def run(self):
        """Generate and display statistics report"""
        if not self.agent.statistics:
            print("No statistics available yet...")
            return
        
        print("\n" + "="*60)
        print(f"TRAFFIC STATISTICS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runtime: {datetime.now() - self.agent.start_time}")
        print("="*60)
        
        total_queue = 0
        total_waiting = 0
        
        for tls_id, stats in self.agent.statistics.items():
            if stats['queue_history']:
                avg_queue = np.mean(stats['queue_history'][-10:])  # Last 10 readings
                avg_waiting = np.mean(stats['waiting_history'][-10:])
                
                print(f"\nIntersection {tls_id}:")
                print(f"  Average Queue Length: {avg_queue:.2f} vehicles")
                print(f"  Average Waiting Time: {avg_waiting:.2f} seconds")
                print(f"  Current Queue: {stats['queue_history'][-1]:.0f} vehicles")
                print(f"  Last Update: {stats['last_update'].strftime('%H:%M:%S')}")
                
                total_queue += avg_queue
                total_waiting += avg_waiting
        
        # Global statistics
        print("\n" + "-"*60)
        print("GLOBAL STATISTICS:")
        print(f"  Total Average Queue: {total_queue:.2f} vehicles")
        print(f"  Total Average Waiting: {total_waiting:.2f} seconds")
        print(f"  Active Intersections: {len(self.agent.statistics)}")
        print("="*60 + "\n")
    
    async def on_start(self):
        """Called when behaviour starts"""
        print("Coordinator started reporting statistics")

    async def on_end(self):
        """Called when behaviour ends"""
        print(f"Coordinator stopped reporting statistics") 