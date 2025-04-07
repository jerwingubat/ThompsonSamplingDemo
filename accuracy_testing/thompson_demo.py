import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import time

class ThompsonSampling:
    def __init__(self, n_machines):
        self.n_machines = n_machines
        self.alpha = np.ones(n_machines)
        self.beta = np.ones(n_machines) 

    def select_machine(self):
        sampled_theta = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_theta), sampled_theta

    def update(self, chosen_machine, reward):
        if reward == 1:
            self.alpha[chosen_machine] += 1
        else:
            self.beta[chosen_machine] += 1


def plot_results(machine_selection_counts, trial, n_machines, alpha, beta_params, sampled_theta, suggested_machine, cumulative_rewards, optimal_rewards):
    plt.clf()

    fig, axs = plt.subplots(2, 2, figsize=(16, 12), num="Thompson Sampling Simulation")


    axs[0, 0].bar(range(n_machines), machine_selection_counts, color='skyblue')
    axs[0, 0].set_title(f"Machine Selections After {trial} Trials")
    axs[0, 0].set_xlabel("Machine")
    axs[0, 0].set_ylabel("Selection Count")
    axs[0, 0].set_xticks(range(n_machines))

    x = np.linspace(0, 1, 100)
    for i in range(n_machines):
        y = beta.pdf(x, alpha[i], beta_params[i])
        axs[0, 1].plot(x, y, label=f"Machine {i}")
        axs[0, 1].scatter([sampled_theta[i]], [beta.pdf(sampled_theta[i], alpha[i], beta_params[i])],
                          color='red', label=f"Sampled {i}" if trial == 1 else None)
        if i == suggested_machine:
            axs[0, 1].axvline(sampled_theta[i], color='orange', linestyle='--', label=f"Suggested Machine {i}")

    axs[0, 1].set_title("Beta Distributions with Sampled Points")
    axs[0, 1].set_xlabel("Probability of Success")
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].legend()

    axs[1, 0].plot(range(1, trial + 1), cumulative_rewards, label="Cumulative Reward", color="blue")
    axs[1, 0].plot(range(1, trial + 1), optimal_rewards, label="Optimal Reward", color="green", linestyle="--")
    axs[1, 0].set_title("Cumulative Rewards vs Optimal Rewards")
    axs[1, 0].set_xlabel("Trials")
    axs[1, 0].set_ylabel("Rewards")
    axs[1, 0].legend()

    # Regret
    regret = np.array(optimal_rewards) - np.array(cumulative_rewards)
    axs[1, 1].plot(range(1, trial + 1), regret, label="Regret", color="red")
    axs[1, 1].set_title("Cumulative Regret Over Time")
    axs[1, 1].set_xlabel("Trials")
    axs[1, 1].set_ylabel("Regret")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.pause(0.001)


def simulate_with_accuracy_and_regret():
    n_machines = 2
    true_conversion_rates = [0.6, 0.2]
    ts = ThompsonSampling(n_machines)
    machine_selection_counts = np.zeros(n_machines, dtype=int)

    cumulative_rewards = []
    optimal_rewards = []
    total_rewards = 0
    max_reward_possible = 0
    n_trials = 100

    print("\n--- Thompson Sampling Simulation ---")
    print(f"True Conversion Rates: {true_conversion_rates}\n")

    for trial in range(1, n_trials + 1):
        suggested_machine, sampled_theta = ts.select_machine()
        chosen_machine = suggested_machine
        machine_selection_counts[chosen_machine] += 1

        reward = 1 if np.random.rand() < true_conversion_rates[chosen_machine] else 0
        total_rewards += reward
        max_reward_possible += max(true_conversion_rates)

        ts.update(chosen_machine, reward)
        cumulative_rewards.append(total_rewards)
        optimal_rewards.append(max_reward_possible)

        plot_results(machine_selection_counts, trial, n_machines, ts.alpha, ts.beta, sampled_theta, suggested_machine, cumulative_rewards, optimal_rewards)
        print(f"Trial {trial}:")
        print(f"  Suggested Machine: {suggested_machine}")
        print(f"  Chosen Machine: {chosen_machine}")
        print(f"  Reward: {reward}")
        print(f"  Total Rewards: {total_rewards}")
        print(f"  Max Possible Rewards: {max_reward_possible}")
        print(f"  Regret: {max_reward_possible - total_rewards:.2f}\n")

        time.sleep(0.2)

    plt.show()

    accuracy = total_rewards / max_reward_possible * 100
    print(f"\nSimulation completed after {n_trials} trials.")
    print(f"Total Rewards Collected: {total_rewards}")
    print(f"Maximum Possible Rewards: {max_reward_possible}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Regret: {max_reward_possible - total_rewards:.2f}\n")

    print("Final Machine Parameters:")
    for i in range(n_machines):
        print(f"Machine {i}: Alpha (success) = {ts.alpha[i]}, Beta (failure) = {ts.beta[i]}")
    print(f"Machine Selection Counts: {machine_selection_counts}")


simulate_with_accuracy_and_regret()
