import os
import pickle
import numpy as np

# Check if Q-tables are being loaded properly
print("Checking Q-tables...")
q_table_path = os.path.join("models", "q_tables.pkl")

if not os.path.exists(q_table_path):
    print(f"ERROR: Q-table file not found: {q_table_path}")
else:
    try:
        with open(q_table_path, 'rb') as f:
            q_tables = pickle.load(f)
        
        print(f"Q-tables loaded successfully. Contains {len(q_tables)} traffic lights.")
        
        # List all traffic lights and sample states
        for tls_id, q_table in q_tables.items():
            print(f"\nTraffic Light {tls_id}:")
            print(f"  States: {len(q_table)}")
            
            # Show some sample states and decisions
            print("  Sample state-action pairs:")
            count = 0
            for state, q_values in q_table.items():
                action = np.argmax(q_values)
                print(f"    State {state}: Action {action} (keep phase)" if action == 0 
                      else f"    State {state}: Action {action} (switch phase)")
                count += 1
                if count >= 5:  # Show max 5 examples
                    break
                    
            # Print action distribution - how many states lead to action 0 vs action 1
            actions = [np.argmax(q_values) for q_values in q_table.values()]
            action_0_count = sum(1 for a in actions if a == 0)
            action_1_count = sum(1 for a in actions if a == 1)
            print(f"  Action distribution: {action_0_count} keep phase ({action_0_count/len(q_table)*100:.1f}%), "
                  f"{action_1_count} switch phase ({action_1_count/len(q_table)*100:.1f}%)")
            
    except Exception as e:
        print(f"Error loading Q-tables: {e}")

# Check if the decision-making logic will ever choose to switch phases
print("\nChecking decision logic...")
print("For a Q-table to be effective:")
print("1. It should have a mix of 'keep phase' and 'switch phase' actions")
print("2. For each traffic light, at least some states should trigger action 1 (switch phase)")
print("3. The state space should cover various traffic conditions")

# Summary
print("\nPossible problems if there's no improvement:")
print("1. Agent decisions not affecting traffic lights (no setPhase calls executing)")
print("2. Q-tables don't have meaningful policies (all states lead to same action)")
print("3. State encoding doesn't capture relevant traffic information")
print("4. Traffic patterns may be too similar in baseline and RL simulations") 