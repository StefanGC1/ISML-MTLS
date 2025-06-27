import numpy as np
import collections
import random
import os
import pickle
import sys
import traci
from sumo_rl import SumoEnvironment

def discretize_queue(queue_length, bins=5):
    """Discretize continuous queue length into bins"""
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

def extract_queue_lengths(obs_data):
    """Extract queue lengths from various observation formats"""
    queue_lengths = []
    
    if isinstance(obs_data, (list, tuple, np.ndarray)):
        # Direct array of values
        queue_lengths = list(obs_data)
    elif isinstance(obs_data, dict):
        # Dictionary format - extract numeric values
        for key, value in obs_data.items():
            if isinstance(value, (int, float)):
                queue_lengths.append(value)
            elif isinstance(value, (list, tuple, np.ndarray)):
                queue_lengths.extend(value)
    else:
        # Single value
        try:
            queue_lengths = [float(obs_data)]
        except:
            print(f"Warning: Cannot extract queue lengths from {type(obs_data)}")
    
    return queue_lengths

def get_state_from_obs(obs, tls_id):
    """Get discretized state from observation"""
    if tls_id not in obs:
        return (0, 0, 0, 0)  # Default state
    
    tls_obs = obs[tls_id]
    queue_lengths = extract_queue_lengths(tls_obs)
    
    # Ensure we have at least 4 values (pad with zeros if needed)
    while len(queue_lengths) < 4:
        queue_lengths.append(0)
    
    # Take first 4 queue lengths and discretize
    state = tuple(discretize_queue(q) for q in queue_lengths[:4])
    return state

def get_traffic_metrics(tls_id):
    """Get current traffic metrics for reward calculation"""
    try:
        # Get controlled lanes
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = sorted(set(controlled_lanes))
        
        metrics = {}
        queues = []
        total_waiting = 0
        vehicles_passed = 0
        
        for lane in unique_lanes:
            # Queue length (halting vehicles)
            queue_length = traci.lane.getLastStepHaltingNumber(lane)
            queues.append(queue_length)
            
            # Waiting time (accumulated)
            waiting_time = traci.lane.getWaitingTime(lane)
            total_waiting += waiting_time
            
            # Vehicles that passed through (flow)
            passed = traci.lane.getLastStepVehicleNumber(lane)
            vehicles_passed += passed
        
        metrics['queues'] = queues
        metrics['total_waiting_time'] = total_waiting
        metrics['vehicles_passed'] = vehicles_passed
        metrics['total_queue'] = sum(queues)
        
        return metrics
    except Exception as e:
        # Return default metrics if error
        return {
            'queues': [0, 0, 0, 0],
            'total_waiting_time': 0,
            'vehicles_passed': 0,
            'total_queue': 0
        }

def calculate_enhanced_reward(tls_id, prev_metrics, current_metrics, action_taken, basic_reward):
    """Calculate multi-component reward for traffic light with improved logic (no switch penalty)"""
    
    # Weights for different components (rebalanced without switch penalty)
    QUEUE_WEIGHT = 1.0      # Penalize long queues (foundation from sumo-rl)
    WAITING_WEIGHT = 0.4    # Increased - penalize waiting time
    EFFICIENCY_WEIGHT = 0.3 # Increased - reward traffic flow
    
    # Component 1: Use basic queue reward as foundation
    queue_component = basic_reward
    
    # Component 2: Improved waiting time penalty (considers both increases and absolute waiting)
    current_waiting = current_metrics.get('total_waiting_time', 0)
    prev_waiting = prev_metrics.get('total_waiting_time', 0)
    waiting_increase = max(0, current_waiting - prev_waiting)
    absolute_waiting = current_waiting / 100.0  # Scale down absolute waiting
    waiting_component = -WAITING_WEIGHT * (waiting_increase / 10.0 + absolute_waiting * 0.1)
    
    # Component 3: Improved efficiency bonus (better flow measurement)
    efficiency_component = 0.0
    try:
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(controlled_lanes))
        
        total_flow_score = 0
        for lane in unique_lanes:
            # Better flow measurement: combine speed and vehicle count
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane)
            
            # Flow score: reward good speed when vehicles are present
            if vehicle_count > 0:
                flow_score = mean_speed * min(vehicle_count / 5.0, 1.0)  # Cap vehicle bonus
                total_flow_score += max(0, flow_score)
        
        if unique_lanes:
            efficiency_component = EFFICIENCY_WEIGHT * (total_flow_score / len(unique_lanes)) / 10.0
            
    except Exception:
        # Fallback to previous simple method if TRACI calls fail
        current_passed = current_metrics.get('vehicles_passed', 0)
        prev_passed = prev_metrics.get('vehicles_passed', 0)
        vehicles_delta = max(0, current_passed - prev_passed)
        efficiency_component = EFFICIENCY_WEIGHT * vehicles_delta * 0.1
    
    # Combine all components (no switch penalty)
    enhanced_reward = (queue_component + 
                      waiting_component + 
                      efficiency_component)
    
    return enhanced_reward

def train_q_learning(episodes=600):
    """Train Q-tables for traffic lights using tabular Q-learning"""
    
    print("Initializing SUMO environment...")
    
    # Try to create environment with flexible parameters
    env_params = {
        'net_file': '../nets/test.net.xml',
        'route_file': '../nets/routes.rou.xml',
        'use_gui': False,
        'single_agent': False,
        'sumo_seed': 42,
    }
    
    # Try different parameter combinations
    try:
        env = SumoEnvironment(
            **env_params,
            reward_fn='queue',
            num_seconds=1800,
            min_green=5,
            max_green=50,
            yellow_time=3)
        print("SumoEnvironment setup first option!")
    except:
        try:
            env = SumoEnvironment(**env_params, seconds=1800)
        except:
            env = SumoEnvironment(**env_params)
    
    print("Environment created successfully!")
    
    # Get traffic light IDs
    try:
        # Reset once to get TLS info
        initial_obs = env.reset()
        if isinstance(initial_obs, tuple):
            initial_obs = initial_obs[0]
        
        tls_ids = list(initial_obs.keys())
        print(f"Traffic lights: {tls_ids}")
    except Exception as e:
        print(f"Error getting traffic lights: {e}")
        # Fallback to expected TLS IDs
        tls_ids = ['J7', 'J9', 'J11', 'J13']
        print(f"Using default traffic lights: {tls_ids}")
    
    # Initialize Q-tables
    Q_tables = {}
    for tls_id in tls_ids:
        Q_tables[tls_id] = collections.defaultdict(lambda: np.zeros(2))
    
    # Q-learning parameters
    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.998
    epsilon_min = 0.01
    
    # Training loop
    episode_rewards = []
    
    print(f"\nStarting training for {episodes} episodes...")
    print("Using enhanced reward function with components:")
    print("  - Queue penalty (from sumo-rl)")
    print("  - Improved waiting time penalty (increases + absolute)")
    print("  - Improved efficiency bonus (speed-based flow measurement)")
    
    for episode in range(episodes):
        # Reset environment
        try:
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result
        except Exception as e:
            print(f"Error resetting environment: {e}")
            break
        
        done = False
        episode_reward = 0
        step_count = 0
        max_steps = 1800  # Prevent infinite loops
        
        # Get initial states and metrics
        states = {}
        previous_metrics = {}
        for tls_id in tls_ids:
            states[tls_id] = get_state_from_obs(obs, tls_id)
            previous_metrics[tls_id] = get_traffic_metrics(tls_id)
        
        while not done and step_count < max_steps:
            # Select actions
            actions = {}
            for tls_id in tls_ids:
                state = states[tls_id]
                
                # Epsilon-greedy
                if random.random() < epsilon:
                    action = random.randint(0, 1)
                else:
                    action = np.argmax(Q_tables[tls_id][state])
                
                actions[tls_id] = action
            
            # Step environment
            try:
                step_result = env.step(actions)
                
                # Handle different return formats
                if len(step_result) >= 3:
                    next_obs = step_result[0]
                    rewards = step_result[1]
                    dones = step_result[2]
                else:
                    print(f"Unexpected step result format: {len(step_result)} values")
                    break
                
                # Check done condition
                if isinstance(dones, dict):
                    done = dones.get('__all__', False)
                else:
                    done = bool(dones)
                
            except Exception as e:
                print(f"Error during step: {e}")
                break
            
            # Update Q-tables with enhanced rewards
            for tls_id in tls_ids:
                state = states[tls_id]
                action = actions[tls_id]
                
                # Get basic reward from environment
                if isinstance(rewards, dict):
                    basic_reward = rewards.get(tls_id, 0)
                else:
                    basic_reward = rewards
                
                # Get current traffic metrics for enhanced reward
                current_metrics = get_traffic_metrics(tls_id)
                prev_metrics = previous_metrics.get(tls_id, {})
                
                # Calculate enhanced reward
                enhanced_reward = calculate_enhanced_reward(
                    tls_id, prev_metrics, current_metrics, action, basic_reward
                )
                
                # Get next state
                next_state = get_state_from_obs(next_obs, tls_id)
                
                # Q-learning update with enhanced reward
                Q_tables[tls_id][state][action] += alpha * (
                    enhanced_reward + gamma * np.max(Q_tables[tls_id][next_state]) - Q_tables[tls_id][state][action]
                )
                
                # Update state and metrics for next iteration
                states[tls_id] = next_state
                previous_metrics[tls_id] = current_metrics
            
            # Track enhanced rewards for episode statistics
            enhanced_episode_reward = 0
            for tls_id in tls_ids:
                current_metrics = get_traffic_metrics(tls_id)
                prev_metrics = previous_metrics.get(tls_id, {})
                
                if isinstance(rewards, dict):
                    basic_reward = rewards.get(tls_id, 0)
                else:
                    basic_reward = rewards
                    
                enhanced_reward = calculate_enhanced_reward(
                    tls_id, prev_metrics, current_metrics, actions[tls_id], basic_reward
                )
                enhanced_episode_reward += enhanced_reward
            
            episode_reward += enhanced_episode_reward
            
            step_count += 1
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if (episode == 300):
            epsilon_decay = 0.997
        elif episode == 400:
            epsilon_decay = 0.996
        elif episode == 500:
            epsilon_decay = 0.995
        
        episode_rewards.append(episode_reward)
        
        # Progress report
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{episodes}, Avg Enhanced Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}, Steps: {step_count}")
    
    # Close environment
    try:
        env.close()
    except:
        pass
    
    # Save Q-tables
    os.makedirs('../models', exist_ok=True)
    
    # Convert to regular dicts
    q_tables_to_save = {}
    for tls_id, q_table in Q_tables.items():
        q_tables_to_save[tls_id] = dict(q_table)
    
    # Save pickle
    with open('../models/q_tables.pkl', 'wb') as f:
        pickle.dump(q_tables_to_save, f)
    
    # Save individual numpy files
    for tls_id, q_table in Q_tables.items():
        states_list = []
        q_values_list = []
        for state, q_values in q_table.items():
            states_list.append(state)
            q_values_list.append(q_values)
        
        if states_list:
            np.savez(f'../models/qtable_{tls_id}.npz', 
                     states=np.array(states_list),
                     q_values=np.array(q_values_list))
    
    print(f"\nTraining complete!")
    print(f"Q-tables saved to ../models/")
    print(f"Final average reward: {np.mean(episode_rewards[-50:]) if episode_rewards else 0:.2f}")
    print(f"Total states learned: {sum(len(qt) for qt in Q_tables.values())}")

if __name__ == "__main__":
    try:
        train_q_learning(episodes=600)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc() 