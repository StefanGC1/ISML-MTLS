#!/usr/bin/env python3
"""
Test script to verify that the environment is properly set up for the Traffic MAS project
"""

import os
import sys
import subprocess
import importlib.util

def test_python_version():
    """Test if Python version is adequate"""
    print("Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is too old. Need Python 3.8+")
        return False

def test_sumo():
    """Test if SUMO is properly installed"""
    print("\nTesting SUMO installation...")
    
    # Check SUMO_HOME environment variable
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        print("✗ SUMO_HOME environment variable not set")
        return False
    
    print(f"✓ SUMO_HOME = {sumo_home}")
    
    # Check if SUMO binary exists
    try:
        result = subprocess.run(['sumo', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.strip().split('\n')[0]
            print(f"✓ {version_line}")
            return True
        else:
            print("✗ SUMO binary not working")
            return False
    except FileNotFoundError:
        print("✗ SUMO binary not found in PATH")
        return False

def test_required_packages():
    """Test if required Python packages are installed"""
    print("\nTesting Python packages...")
    
    required_packages = [
        'spade',
        'traci', 
        'sumo_rl',
        'numpy',
        'gymnasium',
        'aiohttp'
    ]
    
    all_good = True
    for package in required_packages:
        try:
            if package == 'traci':
                # traci is part of SUMO tools
                sys.path.append(os.path.join(os.environ.get('SUMO_HOME', ''), 'tools'))
                import traci
                print(f"✓ {package} available")
            else:
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    print(f"✓ {package} installed")
                else:
                    print(f"✗ {package} not found")
                    all_good = False
        except ImportError:
            print(f"✗ {package} import failed")
            all_good = False
    
    return all_good

def test_network_files():
    """Test if SUMO network files exist"""
    print("\nTesting SUMO network files...")
    
    required_files = [
        'nets/test.net.xml',
        'nets/routes.rou.xml',
        'nets/test.sumocfg'
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_good = False
    
    return all_good

def test_spade_agents():
    """Test if SPADE agents can be imported"""
    print("\nTesting SPADE agents...")
    
    try:
        from agents.intersection import IntersectionAgent
        from agents.coordinator import CoordinatorAgent
        print("✓ Agent classes can be imported")
        return True
    except ImportError as e:
        print(f"✗ Agent import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("Traffic MAS - Environment Test")
    print("="*60)
    
    tests = [
        ("Python Version", test_python_version),
        ("SUMO Installation", test_sumo),
        ("Python Packages", test_required_packages),
        ("Network Files", test_network_files),
        ("SPADE Agents", test_spade_agents)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 Environment is ready! You can now:")
        print("1. Train agents: cd train && python train_qlearn.py")
        print("2. Run simulation: python run_sim.py --duration 600")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        
        print("\nCommon fixes:")
        print("- Install SUMO and set SUMO_HOME environment variable")
        print("- Install Python packages: pip install -r requirements.txt")
        print("- Make sure you're in the project root directory")
    
    print("="*60)

if __name__ == "__main__":
    main() 