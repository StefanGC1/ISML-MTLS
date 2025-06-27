"""
Test script to check sumo-rl version and API
"""

import sys
import pkg_resources

try:
    # Check sumo-rl version
    version = pkg_resources.get_distribution("sumo-rl").version
    print(f"sumo-rl version: {version}")
except:
    print("sumo-rl version: unknown")

try:
    import gymnasium
    gym_version = pkg_resources.get_distribution("gymnasium").version
    print(f"gymnasium version: {gym_version}")
except:
    try:
        import gym
        gym_version = pkg_resources.get_distribution("gym").version
        print(f"gym version: {gym_version}")
    except:
        print("No gym/gymnasium found")

# Test basic environment creation
print("\nTesting environment creation...")

try:
    from sumo_rl import SumoEnvironment
    
    env = SumoEnvironment(
        net_file='../nets/test.net.xml',
        route_file='../nets/routes.rou.xml',
        use_gui=False,
        single_agent=False,
        sumo_seed=42,
        num_seconds=100  # Short episode for testing
    )
    
    print("✓ Environment created successfully")
    
    # Test reset
    print("\nTesting reset...")
    reset_result = env.reset()
    print(f"Reset returns: {type(reset_result)}")
    
    if isinstance(reset_result, tuple):
        print(f"  - Tuple length: {len(reset_result)}")
        obs = reset_result[0]
        print(f"  - Observation type: {type(obs)}")
        if len(reset_result) > 1:
            print(f"  - Info type: {type(reset_result[1])}")
    else:
        obs = reset_result
        print(f"  - Direct observation type: {type(obs)}")
    
    if isinstance(obs, dict):
        print(f"  - Traffic lights: {list(obs.keys())}")
        if obs:
            first_tls = list(obs.keys())[0]
            print(f"  - Observation for {first_tls}: {type(obs[first_tls])}")
            print(f"    Value: {obs[first_tls]}")
    
    # Test step
    print("\nTesting step...")
    if isinstance(obs, dict):
        # Create dummy actions
        actions = {tls: 0 for tls in obs.keys()}
        step_result = env.step(actions)
        print(f"Step returns: {type(step_result)}")
        print(f"  - Length: {len(step_result)}")
        
        if len(step_result) >= 3:
            print(f"  - Next obs type: {type(step_result[0])}")
            print(f"  - Rewards type: {type(step_result[1])}")
            print(f"  - Dones type: {type(step_result[2])}")
            
            if isinstance(step_result[1], dict):
                print(f"    Reward keys: {list(step_result[1].keys())}")
            if isinstance(step_result[2], dict):
                print(f"    Done keys: {list(step_result[2].keys())}")
    
    env.close()
    print("\n✓ All tests passed!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nRecommendation:")
print("If you're having version issues, try:")
print("  pip install --upgrade sumo-rl")
print("  pip install --upgrade gymnasium") 