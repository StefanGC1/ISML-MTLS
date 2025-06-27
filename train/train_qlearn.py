import numpy as np
import collections
import random
import os
import pickle
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

def get_discretized_state(obs, tls_id):
    """Convert observation to discretized state"""
    # obs is a dict with traffic light observations
    # Each TLS has queue lengths for each direction
    if tls_id in obs:
        tls_obs = obs[tls_id]
        # Handle different observation formats
        if isinstance(tls_obs, (list, tuple)):
            # Direct list of queue lengths
            state = tuple(discretize_queue(q) for q in tls_obs)
        elif isinstance(tls_obs, dict):
            # Dictionary with lane information
            queue_lengths = []
            for lane_data in tls_obs.values():
                if isinstance(lane_data, (int, float)):
                    queue_lengths.append(lane_data)
                elif isinstance(lane_data, dict) and 'queue' in lane_data:
                    queue_lengths.append(lane_data['queue'])
            state = tuple(discretize_queue(q) for q in queue_lengths)
        else:
            # Fallback - try to iterate
            try:
                state = tuple(discretize_queue(q) for q in tls_obs)
            except:
                print(f"Warning: Unknown observation format for {tls_id}: {type(tls_obs)}")
                state = tuple()
        return state
    return tuple()

def train_q_learning(episodes=600):
    """Train Q-tables for traffic lights using tabular Q-learning"""
    
    # Initialize environment
    try:
        env = SumoEnvironment(
            net_file='../nets/test.net.xml',
            route_file='../nets/routes.rou.xml',
            num_seconds=1800,  # 30 minutes per episode
            use_gui=False,
            single_agent=False,
            reward_fn='queue',  # Use negative queue length as reward
            sumo_seed=42,
            fixed_ts=False,
            min_green=5,
            max_green=50,
            yellow_time=3
        )
    except TypeError as e:
        # Handle different parameter names in different versions
        print("Trying alternative environment parameters...")
        env = SumoEnvironment(
            net_file='../nets/test.net.xml',
            route_file='../nets/routes.rou.xml',
            seconds=1800,  # Alternative parameter name
            gui=False,
            single_agent=False,
            reward_fn='queue',
            sumo_seed=42,
            fixed_ts=False
        )
    
    # Get traffic light IDs
    tls_ids = list(env.traffic_signals.keys())
    print(f"Traffic lights: {tls_ids}")
    
    # Initialize Q-tables for each traffic light
    Q_tables = {}
    for tls_id in tls_ids:
        Q_tables[tls_id] = collections.defaultdict(lambda: np.zeros(2))  # 2 actions: keep phase or switch
    
    # Q-learning parameters
    alpha = 0.1  # Learning rate
    gamma = 0.95  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.994  # Slightly lower for more exploration with 600 episodes
    epsilon_min = 0.01
    
    # Training loop
    episode_rewards = []
    
    # Debug first reset
    print(f"\nStarting training for {episodes} episodes...")
    
    for episode in range(episodes):
        # Handle different versions of reset()
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            obs = reset_output[0]
        else:
            obs = reset_output
            
        # Debug observation structure (first episode only)
        if episode == 0:
            print(f"Observation type: {type(obs)}")
            if isinstance(obs, dict) and tls_ids:
                print(f"Observation keys: {list(obs.keys())}")
                first_tls = tls_ids[0]
                if first_tls in obs:
                    print(f"Observation for {first_tls}: type={type(obs[first_tls])}, value={obs[first_tls]}")
            
        done = False
        episode_reward = 0
        step = 0
        
        # Get initial states for each TLS
        states = {}
        for tls_id in tls_ids:
            states[tls_id] = get_discretized_state(obs, tls_id)
        
        while not done:
            # Select actions for each traffic light
            actions = {}
            for tls_id in tls_ids:
                state = states[tls_id]
                
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.randint(0, 1)  # Random action
                else:
                    action = np.argmax(Q_tables[tls_id][state])  # Greedy action
                
                actions[tls_id] = action
            
            # Execute actions - handle different versions of step()
            step_output = env.step(actions)
            if len(step_output) == 5:
                next_obs, rewards, dones, _, _ = step_output
            elif len(step_output) == 4:
                next_obs, rewards, dones, _ = step_output
            else:
                next_obs, rewards, dones = step_output[:3]
            
            # Update Q-tables
            for tls_id in tls_ids:
                state = states[tls_id]
                action = actions[tls_id]
                reward = rewards[tls_id]
                next_state = get_discretized_state(next_obs, tls_id)
                
                # Q-learning update
                Q_tables[tls_id][state][action] += alpha * (
                    reward + gamma * np.max(Q_tables[tls_id][next_state]) - Q_tables[tls_id][state][action]
                )
                
                states[tls_id] = next_state
            
            # Track total reward
            episode_reward += sum(rewards.values())
            step += 1
            
            # Check if all agents are done
            if isinstance(dones, dict):
                done = dones.get('__all__', False)
            else:
                done = dones  # Single boolean value
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
    
    env.close()
    
    # Save Q-tables
    os.makedirs('../models', exist_ok=True)
    
    # Convert defaultdicts to regular dicts for saving
    q_tables_to_save = {}
    for tls_id, q_table in Q_tables.items():
        q_tables_to_save[tls_id] = dict(q_table)
    
    with open('../models/q_tables.pkl', 'wb') as f:
        pickle.dump(q_tables_to_save, f)
    
    # Also save as numpy arrays for easier inspection
    for tls_id, q_table in Q_tables.items():
        # Extract all states and Q-values
        states_list = []
        q_values_list = []
        for state, q_values in q_table.items():
            states_list.append(state)
            q_values_list.append(q_values)
        
        if states_list:
            np.savez(f'../models/qtable_{tls_id}.npz', 
                     states=np.array(states_list),
                     q_values=np.array(q_values_list))
    
    print(f"\nTraining complete! Q-tables saved to ../models/")
    print(f"Final average reward: {np.mean(episode_rewards[-50:]):.2f}")

if __name__ == "__main__":
    train_q_learning(episodes=600) 