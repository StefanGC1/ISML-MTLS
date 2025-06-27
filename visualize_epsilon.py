import matplotlib.pyplot as plt
import numpy as np

# Parameters
episodes_current = 400
episodes_proposed = 600
epsilon_decay_current = 0.995
epsilon_decay_proposed = 0.996
epsilon_start = 1.0
epsilon_min = 0.01

# Calculate epsilon values
eps_current = np.arange(episodes_current)
eps_proposed = np.arange(episodes_proposed)

epsilon_values_current = np.maximum(epsilon_min, epsilon_start * (epsilon_decay_current ** eps_current))
epsilon_values_proposed = np.maximum(epsilon_min, epsilon_start * (epsilon_decay_proposed ** eps_proposed))

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(eps_current, epsilon_values_current, 'b-', label=f'Current: 400 eps, decay=0.995', linewidth=2)
plt.plot(eps_proposed, epsilon_values_proposed, 'r--', label=f'Proposed: 600 eps, decay=0.994', linewidth=2)
plt.axhline(y=epsilon_min, color='gray', linestyle=':', label=f'Min epsilon = {epsilon_min}')

# Highlight key points
plt.scatter([400], [epsilon_values_current[-1]], color='blue', s=100, zorder=5)
plt.scatter([600], [epsilon_values_proposed[-1]], color='red', s=100, zorder=5)

plt.xlabel('Episode')
plt.ylabel('Epsilon (Exploration Rate)')
plt.title('Epsilon Decay Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 600)
plt.ylim(0, 1.05)

# Add annotations
plt.annotate(f'ε={epsilon_values_current[-1]:.3f}', 
             xy=(400, epsilon_values_current[-1]), 
             xytext=(420, epsilon_values_current[-1] + 0.05),
             arrowprops=dict(arrowstyle='->', color='blue'))
plt.annotate(f'ε={epsilon_values_proposed[-1]:.3f}', 
             xy=(600, epsilon_values_proposed[-1]), 
             xytext=(500, epsilon_values_proposed[-1] + 0.1),
             arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('epsilon_decay_comparison.png', dpi=150)
print("Plot saved as epsilon_decay_comparison.png")

# Print comparison
print("\nEpsilon values at key episodes:")
print("Episode | Current (0.995) | Proposed (0.994)")
print("--------|-----------------|------------------")
for ep in [100, 200, 300, 400, 500, 600]:
    if ep <= 400:
        current = epsilon_start * (epsilon_decay_current ** ep)
        proposed = epsilon_start * (epsilon_decay_proposed ** ep)
        print(f"{ep:7d} | {current:15.3f} | {proposed:16.3f}")
    else:
        proposed = epsilon_start * (epsilon_decay_proposed ** ep)
        print(f"{ep:7d} | {'N/A':>15s} | {proposed:16.3f}")