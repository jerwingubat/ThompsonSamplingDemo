import numpy as np
import matplotlib.pyplot as plt

class ThompsonSampling:
    def __init__(self, n_machaines):
        self.n_machines = n_machaines
        self.alpha = np.ones(n_machines)
        self.beta = np.ones(n_machines)

    def select_machine(self):
        sampled_theta = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_theta)

    def update(self, chosen_machine, reward):
        if reward == 1:
            self.alpha[chosen_machine] += 1
        else:
            self.beta[chosen_machine] += 1


def simulate_thompson_sampling(n_machines, true_probs, n_trials):
    ts = ThompsonSampling(n_machines)
    rewards = np.zeros(n_trials)
    optimal_rewards = np.zeros(n_trials)
    chosen_optimal_machine = 0

    for trial in range(n_trials):
        chosen_machine = ts.select_machine()

        reward = np.random.rand() < true_probs[chosen_machine]
        rewards[trial] = reward
        ts.update(chosen_machine, reward)
        optimal_rewards[trial] = np.max(true_probs)

        if chosen_machine == np.argmax(true_probs):
            chosen_optimal_machine += 1

    cumulative_rewards = np.cumsum(rewards)
    cumulative_optimal_rewards = np.cumsum(optimal_rewards)
    regret = cumulative_optimal_rewards - cumulative_rewards

    return cumulative_rewards, cumulative_optimal_rewards, regret, chosen_optimal_machine

n_machines = 3
true_probs = [0.4, 0.5, 0.6] 
n_trials = 1000

cumulative_rewards, cumulative_optimal_rewards, regret, chosen_optimal_machine = simulate_thompson_sampling(n_machines, true_probs, n_trials)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(cumulative_rewards, label="Thompson Sampling")
plt.plot(cumulative_optimal_rewards, label="Optimal", linestyle="--")
plt.xlabel("Trials")
plt.ylabel("Cumulative Rewards")
plt.title("Cumulative Rewards vs. Optimal")
plt.legend()

# Regret over time
plt.subplot(1, 2, 2)
plt.plot(regret, label="Regret", color="red")
plt.xlabel("Trials")
plt.ylabel("Regret")
plt.title("Regret Over Time")
plt.legend()

plt.tight_layout()
plt.show()
final_cumulative_rewards = cumulative_rewards[-1]
final_cumulative_optimal_rewards = cumulative_optimal_rewards[-1]
final_regret = regret[-1]
accuracy = chosen_optimal_machine / n_trials

print(f"Final Cumulative Rewards: {final_cumulative_rewards}")
print(f"Final Cumulative Optimal Rewards: {final_cumulative_optimal_rewards}")
print(f"Final Regret: {final_regret}")
print(f"Accuracy (Optimal Machine Selection Rate): {accuracy * 100:.2f}%")
