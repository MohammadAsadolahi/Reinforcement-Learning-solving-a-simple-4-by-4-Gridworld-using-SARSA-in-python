"""
Generate publication-quality plots for SARSA GridWorld README.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import os

np.random.seed(42)

# ─── GridWorld and Agent (clean reimplementation for plotting) ───


class GridWorld:
    def __init__(self):
        self.actionSpace = ('U', 'D', 'L', 'R')
        self.actions = {
            (0, 0): ('D', 'R'), (0, 1): ('L', 'D', 'R'), (0, 2): ('L', 'D', 'R'), (0, 3): ('L', 'D'),
            (1, 0): ('U', 'D', 'R'), (1, 1): ('U', 'L', 'D', 'R'), (1, 2): ('U', 'L', 'D', 'R'), (1, 3): ('U', 'L', 'D'),
            (2, 0): ('U', 'D', 'R'), (2, 1): ('U', 'L', 'D', 'R'), (2, 2): ('U', 'L', 'D', 'R'), (2, 3): ('U', 'L', 'D'),
            (3, 0): ('U', 'R'), (3, 1): ('U', 'L', 'R'), (3, 2): ('U', 'L', 'R')
        }
        self.rewards = {(3, 3): 0.5, (1, 3): -0.5, (2, 1): -0.5, (3, 1): -0.5}

    def reset(self):
        self.state = (0, 0)
        return self.state

    def is_terminal(self, s):
        return s not in self.actions

    def move(self, action):
        r, c = self.state
        if action == 'U':
            r -= 1
        elif action == 'D':
            r += 1
        elif action == 'L':
            c -= 1
        elif action == 'R':
            c += 1
        self.state = (r, c)
        reward = self.rewards.get((r, c), -0.01)
        return (r, c), reward, self.is_terminal(self.state)


class Agent:
    def __init__(self, action_space, alpha=0.1, gamma=0.9, exploreRate=0.01):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.exploreRate = exploreRate
        self.qTable = {}
        self.policy = {}
        for state in action_space:
            self.qTable[state] = {a: 0.0 for a in action_space[state]}
            self.policy[state] = np.random.choice(action_space[state])
        self.explored = 0
        self.exploited = 0

    def chooseAction(self, state):
        if self.exploreRate > np.random.rand():
            self.explored += 1
            return np.random.choice(self.action_space[state])
        self.exploited += 1
        return self.policy[state]

    def learn(self, state, action, nextState, reward, done):
        if not done:
            next_action = self.chooseAction(nextState)
            targetQ = reward + self.gamma * self.qTable[nextState][next_action]
        else:
            targetQ = reward
        self.qTable[state][action] += self.alpha * \
            (targetQ - self.qTable[state][action])

    def update_policy(self):
        for state in self.policy:
            self.policy[state] = max(
                self.qTable[state], key=self.qTable[state].get)


# ─── Training with metric collection ───

def train(num_episodes=2000, alpha=0.1, gamma=0.9, exploreRate=0.01):
    env = GridWorld()
    agent = Agent(env.actions, alpha=alpha,
                  gamma=gamma, exploreRate=exploreRate)

    episode_rewards = []
    episode_steps = []
    policy_snapshots = {}

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done and steps < 50:
            action = agent.chooseAction(state)
            next_state, reward, done = env.move(action)
            agent.learn(state, action, next_state, reward, done)
            total_reward += reward
            state = next_state
            steps += 1
        agent.update_policy()
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        if ep in [0, 50, 200, 500, 1000, 1999]:
            policy_snapshots[ep] = dict(agent.policy)

    return agent, episode_rewards, episode_steps, policy_snapshots


agent, rewards, steps, snapshots = train()

# Create assets directory
os.makedirs('assets', exist_ok=True)

# ─── Plot 1: GridWorld Environment Diagram ───

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal')
ax.invert_yaxis()

cell_labels = {
    (0, 0): ('S', '#4CAF50', 'white'),     # Start - green
    (1, 3): ('H', '#F44336', 'white'),      # Hole - red
    (2, 1): ('H', '#F44336', 'white'),      # Hole - red
    (3, 1): ('H', '#F44336', 'white'),      # Hole - red
    (3, 3): ('G', '#2196F3', 'white'),      # Goal - blue
}

for r in range(4):
    for c in range(4):
        if (r, c) in cell_labels:
            label, color, text_color = cell_labels[(r, c)]
        else:
            label, color, text_color = ('', '#E8F5E9', '#333')

        rect = plt.Rectangle((c-0.5, r-0.5), 1, 1, linewidth=2,
                             edgecolor='#37474F', facecolor=color, alpha=0.85)
        ax.add_patch(rect)
        if label:
            ax.text(c, r, label, ha='center', va='center',
                    fontsize=28, fontweight='bold', color=text_color,
                    fontfamily='monospace')

# Reward annotations
reward_labels = {(0, 0): 'Start', (1, 3): 'R=-0.5', (2, 1)
                  : 'R=-0.5', (3, 1): 'R=-0.5', (3, 3): 'R=+0.5'}
for (r, c), txt in reward_labels.items():
    ax.text(c, r+0.35, txt, ha='center', va='center', fontsize=10, color='white' if (r, c) != (0, 0) else 'white',
            fontweight='bold', style='italic')

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(['Col 0', 'Col 1', 'Col 2', 'Col 3'],
                   fontsize=12, fontweight='bold')
ax.set_yticklabels(['Row 0', 'Row 1', 'Row 2', 'Row 3'],
                   fontsize=12, fontweight='bold')
ax.set_title('4×4 GridWorld Environment', fontsize=20,
             fontweight='bold', pad=15, color='#1a237e')

legend_elements = [
    mpatches.Patch(facecolor='#4CAF50', edgecolor='#333', label='Start (S)'),
    mpatches.Patch(facecolor='#F44336', edgecolor='#333',
                   label='Hole (H) — penalty'),
    mpatches.Patch(facecolor='#2196F3', edgecolor='#333',
                   label='Goal (G) — terminal'),
    mpatches.Patch(facecolor='#E8F5E9', edgecolor='#333',
                   label='Safe cell (R=−0.01)'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          ncol=2, fontsize=11, frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('assets/gridworld_env.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()


# ─── Plot 2: Learning Curves ───

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

# Smoothing function


def smooth(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')


# Cumulative reward
ax = axes[0]
smoothed_rewards = smooth(rewards)
ax.plot(smoothed_rewards, color='#1565C0', linewidth=1.8, alpha=0.9)
ax.fill_between(range(len(smoothed_rewards)),
                smoothed_rewards, alpha=0.15, color='#1565C0')
ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Reward (50-ep moving avg)',
              fontsize=12, fontweight='bold')
ax.set_title('Reward Convergence', fontsize=16,
             fontweight='bold', color='#1a237e')
ax.axhline(y=max(smoothed_rewards), color='#4CAF50', linestyle='--',
           alpha=0.7, label=f'Peak = {max(smoothed_rewards):.3f}')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Steps per episode
ax = axes[1]
smoothed_steps = smooth(steps)
ax.plot(smoothed_steps, color='#E65100', linewidth=1.8, alpha=0.9)
ax.fill_between(range(len(smoothed_steps)), smoothed_steps,
                alpha=0.15, color='#E65100')
ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
ax.set_ylabel('Steps per Episode (50-ep moving avg)',
              fontsize=12, fontweight='bold')
ax.set_title('Episode Length Convergence', fontsize=16,
             fontweight='bold', color='#1a237e')
ax.axhline(y=min(smoothed_steps), color='#4CAF50', linestyle='--',
           alpha=0.7, label=f'Min = {min(smoothed_steps):.1f}')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('SARSA Learning Dynamics Over 2,000 Episodes',
             fontsize=18, fontweight='bold', y=1.02, color='#0D47A1')
plt.tight_layout()
plt.savefig('assets/learning_curves.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()


# ─── Plot 3: Q-Value Heatmap ───

fig, ax = plt.subplots(figsize=(9, 8))
q_max = np.zeros((4, 4))
q_max[:] = np.nan

for (r, c), actions in agent.qTable.items():
    q_max[r][c] = max(actions.values())
q_max[3][3] = 0.5  # terminal

cmap = plt.cm.RdYlGn
norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(
    q_max), vcenter=0, vmax=np.nanmax(q_max))
im = ax.imshow(q_max, cmap=cmap, norm=norm, aspect='equal')

for r in range(4):
    for c in range(4):
        if not np.isnan(q_max[r][c]):
            val = q_max[r][c]
            text_color = 'white' if abs(val) > 0.2 else 'black'
            ax.text(c, r, f'{val:.3f}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color=text_color)

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(['Col 0', 'Col 1', 'Col 2', 'Col 3'],
                   fontsize=12, fontweight='bold')
ax.set_yticklabels(['Row 0', 'Row 1', 'Row 2', 'Row 3'],
                   fontsize=12, fontweight='bold')
ax.set_title('Max Q-Values per State (Learned Value Landscape)',
             fontsize=16, fontweight='bold', color='#1a237e', pad=15)
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Q-Value', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/q_value_heatmap.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()


# ─── Plot 4: Learned Optimal Policy with Arrows ───

arrow_map = {'U': (0, -0.3), 'D': (0, 0.3), 'L': (-0.3, 0), 'R': (0.3, 0)}

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal')
ax.invert_yaxis()

for r in range(4):
    for c in range(4):
        if (r, c) in cell_labels:
            _, color, _ = cell_labels[(r, c)]
        else:
            color = '#FAFAFA'
        rect = plt.Rectangle((c-0.5, r-0.5), 1, 1, linewidth=2,
                             edgecolor='#546E7A', facecolor=color, alpha=0.7)
        ax.add_patch(rect)

for state, action in agent.policy.items():
    r, c = state
    dx, dy = arrow_map[action]
    ax.annotate('', xy=(c+dx, r+dy), xytext=(c, r),
                arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=3, mutation_scale=22))

# Mark S and G
ax.text(0, 0, 'S', ha='center', va='center', fontsize=24,
        fontweight='bold', color='white', zorder=5)
ax.text(3, 3, 'G', ha='center', va='center', fontsize=24,
        fontweight='bold', color='white', zorder=5)
for (r, c) in [(1, 3), (2, 1), (3, 1)]:
    ax.text(c, r, 'H', ha='center', va='center', fontsize=24,
            fontweight='bold', color='white', zorder=5)

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(['Col 0', 'Col 1', 'Col 2', 'Col 3'],
                   fontsize=12, fontweight='bold')
ax.set_yticklabels(['Row 0', 'Row 1', 'Row 2', 'Row 3'],
                   fontsize=12, fontweight='bold')
ax.set_title('Learned Optimal Policy (After 2,000 Episodes)',
             fontsize=17, fontweight='bold', pad=15, color='#1a237e')
plt.tight_layout()
plt.savefig('assets/optimal_policy.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()


# ─── Plot 5: Policy Evolution Snapshots ───

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
snap_keys = sorted(snapshots.keys())

for idx, ep in enumerate(snap_keys):
    ax = axes[idx // 3][idx % 3]
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    for r in range(4):
        for c in range(4):
            if (r, c) in cell_labels:
                _, color, _ = cell_labels[(r, c)]
            else:
                color = '#ECEFF1'
            rect = plt.Rectangle((c-0.5, r-0.5), 1, 1, linewidth=1.5,
                                 edgecolor='#78909C', facecolor=color, alpha=0.65)
            ax.add_patch(rect)

    policy = snapshots[ep]
    for state, action in policy.items():
        r, c = state
        dx, dy = arrow_map[action]
        ax.annotate('', xy=(c+dx, r+dy), xytext=(c, r),
                    arrowprops=dict(arrowstyle='->', color='#1A237E', lw=2.5, mutation_scale=18))

    ax.text(0, 0, 'S', ha='center', va='center', fontsize=18,
            fontweight='bold', color='white', zorder=5)
    ax.text(3, 3, 'G', ha='center', va='center', fontsize=18,
            fontweight='bold', color='white', zorder=5)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f'Episode {ep}', fontsize=15,
                 fontweight='bold', color='#283593')

plt.suptitle('Policy Evolution During SARSA Training',
             fontsize=20, fontweight='bold', color='#0D47A1', y=1.01)
plt.tight_layout()
plt.savefig('assets/policy_evolution.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()


# ─── Plot 6: Exploration vs Exploitation Pie ───

fig, ax = plt.subplots(figsize=(7, 7))
labels = ['Exploitation\n(Greedy)', 'Exploration\n(Random)']
sizes = [agent.exploited, agent.explored]
colors = ['#1565C0', '#FF6F00']
explode = (0.03, 0.08)
wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                  autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14, 'fontweight': 'bold'},
                                  wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for t in autotexts:
    t.set_color('white')
    t.set_fontsize(15)
ax.set_title('Exploration vs. Exploitation Ratio', fontsize=17,
             fontweight='bold', color='#1a237e', pad=20)
ax.text(0, -1.35, f'Total actions: {agent.exploited + agent.explored:,}   |   ε = 0.01',
        ha='center', fontsize=12, color='#555', style='italic')
plt.tight_layout()
plt.savefig('assets/explore_exploit.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()


# ─── Plot 7: Optimal Trajectory on Grid ───

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal')
ax.invert_yaxis()

for r in range(4):
    for c in range(4):
        if (r, c) in cell_labels:
            _, color, _ = cell_labels[(r, c)]
        else:
            color = '#F5F5F5'
        rect = plt.Rectangle((c-0.5, r-0.5), 1, 1, linewidth=2,
                             edgecolor='#546E7A', facecolor=color, alpha=0.6)
        ax.add_patch(rect)

# Trace optimal path from S to G
env = GridWorld()
env.reset()
path = [(0, 0)]
done = False
max_steps = 20
s = 0
while not done and s < max_steps:
    action = agent.policy.get(env.state)
    if action is None:
        break
    next_state, _, done = env.move(action)
    path.append(next_state)
    s += 1

# Draw path
path_x = [c for r, c in path]
path_y = [r for r, c in path]
ax.plot(path_x, path_y, 'o-', color='#FF6F00', linewidth=4, markersize=14, markerfacecolor='#FFB300',
        markeredgecolor='#E65100', markeredgewidth=2, zorder=4, alpha=0.9)

for i, (r, c) in enumerate(path):
    ax.text(c, r-0.38, str(i), ha='center', va='center', fontsize=10,
            fontweight='bold', color='#BF360C', zorder=6)

ax.text(0, 0, 'S', ha='center', va='center', fontsize=22,
        fontweight='bold', color='white', zorder=5)
ax.text(3, 3, 'G', ha='center', va='center', fontsize=22,
        fontweight='bold', color='white', zorder=5)
for (r, c) in [(1, 3), (2, 1), (3, 1)]:
    ax.text(c, r, 'H', ha='center', va='center', fontsize=22,
            fontweight='bold', color='white', zorder=5)

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(['Col 0', 'Col 1', 'Col 2', 'Col 3'],
                   fontsize=12, fontweight='bold')
ax.set_yticklabels(['Row 0', 'Row 1', 'Row 2', 'Row 3'],
                   fontsize=12, fontweight='bold')
ax.set_title(f'Optimal Trajectory: {len(path)-1} Steps from Start to Goal',
             fontsize=16, fontweight='bold', pad=15, color='#1a237e')
plt.tight_layout()
plt.savefig('assets/optimal_trajectory.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()

# ─── Print stats for README ───
print("=== STATS ===")
print(f"Final avg reward (last 100): {np.mean(rewards[-100:]):.4f}")
print(f"Final avg steps (last 100): {np.mean(steps[-100:]):.2f}")
print(f"Total exploited: {agent.exploited}")
print(f"Total explored: {agent.explored}")
print(f"Optimal path length: {len(path)-1}")
print(f"Optimal path: {' → '.join(str(p) for p in path)}")
print(f"Final policy: {agent.policy}")
print("Plots saved to assets/")
