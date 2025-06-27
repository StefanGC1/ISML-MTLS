"""
Compare baseline and RL simulation results
Analyzes SUMO output files to calculate improvements
"""

import xml.etree.ElementTree as ET
import os
import sys
import glob
import numpy as np
from datetime import datetime


class SimulationResults:
    """Container for simulation statistics"""
    def __init__(self, name):
        self.name = name
        self.trips = []
        self.summary_intervals = []
        self.final_stats = {}
        
    def parse_tripinfo(self, filename):
        """Parse tripinfo XML file"""
        tree = ET.parse(filename)
        root = tree.getroot()
        
        for trip in root.findall('tripinfo'):
            trip_data = {
                'id': trip.get('id'),
                'depart': float(trip.get('depart')),
                'arrival': float(trip.get('arrival')),
                'duration': float(trip.get('duration')),
                'routeLength': float(trip.get('routeLength')),
                'waitingTime': float(trip.get('waitingTime')),
                'waitingCount': int(trip.get('waitingCount')),
                'stopTime': float(trip.get('stopTime')),
                'timeLoss': float(trip.get('timeLoss')),
                'speed': float(trip.get('routeLength')) / float(trip.get('duration')) if float(trip.get('duration')) > 0 else 0
            }
            self.trips.append(trip_data)
    
    def parse_summary(self, filename):
        """Parse summary XML file"""
        tree = ET.parse(filename)
        root = tree.getroot()
        
        for step in root.findall('step'):
            step_data = {
                'time': float(step.get('time')),
                'loaded': int(step.get('loaded')),
                'inserted': int(step.get('inserted')),
                'running': int(step.get('running')),
                'waiting': int(step.get('waiting')),
                'ended': int(step.get('ended')),
                'meanWaitingTime': float(step.get('meanWaitingTime')),
                'meanTravelTime': float(step.get('meanTravelTime')),
                'halting': int(step.get('halting')),
                'meanSpeed': float(step.get('meanSpeed')),
                'meanSpeedRelative': float(step.get('meanSpeedRelative'))
            }
            self.summary_intervals.append(step_data)
    
    def parse_statistics(self, filename):
        """Parse statistics XML file"""
        tree = ET.parse(filename)
        root = tree.getroot()
        
        # Get vehicle statistics
        vehicles = root.find('vehicles')
        if vehicles is not None:
            self.final_stats['total_vehicles'] = int(vehicles.get('loaded'))
            self.final_stats['total_departed'] = int(vehicles.get('inserted'))
            # Fix: handle missing 'arrived' attribute by calculating from available data
            # arrived = inserted - (running + waiting)
            arrived = vehicles.get('arrived')
            if arrived is not None:
                self.final_stats['total_arrived'] = int(arrived)
            else:
                inserted = int(vehicles.get('inserted', 0))
                running = int(vehicles.get('running', 0))
                waiting = int(vehicles.get('waiting', 0))
                self.final_stats['total_arrived'] = inserted - running - waiting
        
        # Get performance statistics
        perf = root.find('vehicleTripStatistics')
        if perf is not None:
            self.final_stats['avg_duration'] = float(perf.get('duration', 0))
            self.final_stats['avg_waiting_time'] = float(perf.get('waitingTime', 0))
            self.final_stats['avg_time_loss'] = float(perf.get('timeLoss', 0))
            self.final_stats['avg_speed'] = float(perf.get('speed', 0))
            self.final_stats['total_travel_time'] = float(perf.get('totalTravelTime', 0))
    
    def calculate_statistics(self):
        """Calculate aggregate statistics from trips"""
        if not self.trips:
            return
        
        # Calculate statistics from individual trips
        durations = [t['duration'] for t in self.trips]
        waiting_times = [t['waitingTime'] for t in self.trips]
        time_losses = [t['timeLoss'] for t in self.trips]
        speeds = [t['speed'] for t in self.trips]
        
        self.trip_stats = {
            'count': len(self.trips),
            'avg_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'avg_waiting_time': np.mean(waiting_times),
            'std_waiting_time': np.std(waiting_times),
            'total_waiting_time': np.sum(waiting_times),
            'avg_time_loss': np.mean(time_losses),
            'total_time_loss': np.sum(time_losses),
            'avg_speed': np.mean(speeds),
            'vehicles_with_stops': sum(1 for t in self.trips if t['waitingCount'] > 0),
            'avg_stops_per_vehicle': np.mean([t['waitingCount'] for t in self.trips])
        }
        
        # Calculate statistics from summary intervals
        if self.summary_intervals:
            mean_waiting_times = [s['meanWaitingTime'] for s in self.summary_intervals]
            mean_travel_times = [s['meanTravelTime'] for s in self.summary_intervals]
            halting_counts = [s['halting'] for s in self.summary_intervals]
            mean_speeds = [s['meanSpeed'] for s in self.summary_intervals]
            
            self.interval_stats = {
                'avg_mean_waiting_time': np.mean(mean_waiting_times),
                'max_mean_waiting_time': np.max(mean_waiting_times),
                'avg_mean_travel_time': np.mean(mean_travel_times),
                'avg_halting_vehicles': np.mean(halting_counts),
                'max_halting_vehicles': np.max(halting_counts),
                'avg_mean_speed': np.mean(mean_speeds)
            }


def find_latest_files(directory):
    """Find the latest set of output files in a directory"""
    # Find all tripinfo files
    tripinfo_files = glob.glob(os.path.join(directory, "tripinfo_*.xml"))
    if not tripinfo_files:
        return None
    
    # Get the latest file
    latest_tripinfo = max(tripinfo_files, key=os.path.getctime)
    
    # Extract timestamp
    timestamp = os.path.basename(latest_tripinfo).replace("tripinfo_", "").replace(".xml", "")
    
    # Build file paths
    files = {
        'tripinfo': latest_tripinfo,
        'summary': os.path.join(directory, f"summary_{timestamp}.xml"),
        'statistics': os.path.join(directory, f"statistics_{timestamp}.xml"),
        'queue': os.path.join(directory, f"queue_{timestamp}.xml")
    }
    
    # Check if all files exist
    for file_type, path in files.items():
        if not os.path.exists(path):
            print(f"Warning: {file_type} file not found: {path}")
    
    return files, timestamp


def compare_simulations(baseline_dir="output/baseline", rl_dir="output/rl"):
    """Compare baseline and RL simulation results"""
    
    print("="*70)
    print("TRAFFIC SIMULATION COMPARISON")
    print("="*70)
    
    # Find latest files
    baseline_files, baseline_timestamp = find_latest_files(baseline_dir)
    rl_files, rl_timestamp = find_latest_files(rl_dir)
    
    if not baseline_files:
        print("Error: No baseline results found. Run baseline simulation first.")
        return
    
    if not rl_files:
        print("Error: No RL results found. Run RL simulation first.")
        return
    
    print(f"\nBaseline simulation: {baseline_timestamp}")
    print(f"RL simulation:       {rl_timestamp}")
    
    # Load baseline results
    baseline = SimulationResults("Baseline (Fixed-time)")
    if os.path.exists(baseline_files['tripinfo']):
        baseline.parse_tripinfo(baseline_files['tripinfo'])
    if os.path.exists(baseline_files['summary']):
        baseline.parse_summary(baseline_files['summary'])
    if os.path.exists(baseline_files['statistics']):
        baseline.parse_statistics(baseline_files['statistics'])
    baseline.calculate_statistics()
    
    # Load RL results
    rl = SimulationResults("RL-controlled")
    if os.path.exists(rl_files['tripinfo']):
        rl.parse_tripinfo(rl_files['tripinfo'])
    if os.path.exists(rl_files['summary']):
        rl.parse_summary(rl_files['summary'])
    if os.path.exists(rl_files['statistics']):
        rl.parse_statistics(rl_files['statistics'])
    rl.calculate_statistics()
    
    # Compare results
    print("\n" + "="*70)
    print("TRIP STATISTICS (from individual vehicles)")
    print("="*70)
    
    if hasattr(baseline, 'trip_stats') and hasattr(rl, 'trip_stats'):
        metrics = [
            ('Total vehicles', 'count', '{:.0f}'),
            ('Average travel time', 'avg_duration', '{:.1f}s'),
            ('Average waiting time', 'avg_waiting_time', '{:.1f}s'),
            ('Total waiting time', 'total_waiting_time', '{:.1f}s'),
            ('Average time loss', 'avg_time_loss', '{:.1f}s'),
            ('Average speed', 'avg_speed', '{:.1f} m/s'),
            ('Vehicles that stopped', 'vehicles_with_stops', '{:.0f}'),
            ('Avg stops per vehicle', 'avg_stops_per_vehicle', '{:.2f}')
        ]
        
        print(f"{'Metric':<25} {'Baseline':>15} {'RL-controlled':>15} {'Improvement':>15}")
        print("-"*70)
        
        for metric_name, key, fmt in metrics:
            baseline_val = baseline.trip_stats.get(key, 0)
            rl_val = rl.trip_stats.get(key, 0)
            
            if baseline_val > 0:
                if key in ['avg_waiting_time', 'total_waiting_time', 'avg_time_loss', 'avg_duration']:
                    # Lower is better
                    improvement = (baseline_val - rl_val) / baseline_val * 100
                    sign = '+' if improvement > 0 else ''
                elif key in ['avg_speed']:
                    # Higher is better
                    improvement = (rl_val - baseline_val) / baseline_val * 100
                    sign = '+' if improvement > 0 else ''
                else:
                    improvement = 0
                    sign = ''
            else:
                improvement = 0
                sign = ''
            
            print(f"{metric_name:<25} {fmt.format(baseline_val):>15} {fmt.format(rl_val):>15} {sign}{improvement:>13.1f}%")
    
    # Compare interval statistics
    print("\n" + "="*70)
    print("INTERVAL STATISTICS (60-second averages)")
    print("="*70)
    
    if hasattr(baseline, 'interval_stats') and hasattr(rl, 'interval_stats'):
        metrics = [
            ('Avg mean waiting time', 'avg_mean_waiting_time', '{:.1f}s'),
            ('Max mean waiting time', 'max_mean_waiting_time', '{:.1f}s'),
            ('Avg mean travel time', 'avg_mean_travel_time', '{:.1f}s'),
            ('Avg halting vehicles', 'avg_halting_vehicles', '{:.1f}'),
            ('Max halting vehicles', 'max_halting_vehicles', '{:.0f}'),
            ('Avg mean speed', 'avg_mean_speed', '{:.1f} m/s')
        ]
        
        print(f"{'Metric':<25} {'Baseline':>15} {'RL-controlled':>15} {'Improvement':>15}")
        print("-"*70)
        
        for metric_name, key, fmt in metrics:
            baseline_val = baseline.interval_stats.get(key, 0)
            rl_val = rl.interval_stats.get(key, 0)
            
            if baseline_val > 0:
                if key in ['avg_mean_waiting_time', 'max_mean_waiting_time', 'avg_mean_travel_time', 
                          'avg_halting_vehicles', 'max_halting_vehicles']:
                    # Lower is better
                    improvement = (baseline_val - rl_val) / baseline_val * 100
                    sign = '+' if improvement > 0 else ''
                elif key in ['avg_mean_speed']:
                    # Higher is better
                    improvement = (rl_val - baseline_val) / baseline_val * 100
                    sign = '+' if improvement > 0 else ''
                else:
                    improvement = 0
                    sign = ''
            else:
                improvement = 0
                sign = ''
            
            print(f"{metric_name:<25} {fmt.format(baseline_val):>15} {fmt.format(rl_val):>15} {sign}{improvement:>13.1f}%")
    
    # Overall summary
    print("\n" + "="*70)
    print("KEY IMPROVEMENTS")
    print("="*70)
    
    if hasattr(baseline, 'trip_stats') and hasattr(rl, 'trip_stats'):
        # Calculate key improvements
        wait_time_reduction = (baseline.trip_stats['avg_waiting_time'] - rl.trip_stats['avg_waiting_time']) / baseline.trip_stats['avg_waiting_time'] * 100
        travel_time_reduction = (baseline.trip_stats['avg_duration'] - rl.trip_stats['avg_duration']) / baseline.trip_stats['avg_duration'] * 100
        speed_increase = (rl.trip_stats['avg_speed'] - baseline.trip_stats['avg_speed']) / baseline.trip_stats['avg_speed'] * 100
        
        print(f"✓ Waiting time reduced by: {wait_time_reduction:.1f}%")
        print(f"✓ Travel time reduced by: {travel_time_reduction:.1f}%")
        print(f"✓ Average speed increased by: {speed_increase:.1f}%")
        
        total_wait_saved = baseline.trip_stats['total_waiting_time'] - rl.trip_stats['total_waiting_time']
        print(f"✓ Total waiting time saved: {total_wait_saved:.0f} seconds")
        
        if total_wait_saved > 0:
            print(f"  (Equivalent to {total_wait_saved/3600:.1f} vehicle-hours saved)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare baseline and RL simulation results")
    parser.add_argument("--baseline-dir", default="output/baseline", 
                        help="Directory containing baseline results")
    parser.add_argument("--rl-dir", default="output/rl", 
                        help="Directory containing RL results")
    
    args = parser.parse_args()
    
    compare_simulations(args.baseline_dir, args.rl_dir) 