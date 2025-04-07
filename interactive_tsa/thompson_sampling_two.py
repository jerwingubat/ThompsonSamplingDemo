import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

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


def plot_machine_selection_and_distributions(machine_selection_counts, trial, n_machines, alpha, beta_params, sampled_theta, suggested_machine):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].bar(range(n_machines), machine_selection_counts, color='skyblue')
    axs[0].set_title(f"Machine Selections After {trial} Trials")
    axs[0].set_xlabel("Machine")
    axs[0].set_ylabel("Selection Count")
    axs[0].set_xticks(range(n_machines))

    x = np.linspace(0, 1, 100)
    for i in range(n_machines):
        y = beta.pdf(x, alpha[i], beta_params[i])
        axs[1].plot(x, y, label=f"Machine {i}")
        axs[1].scatter([sampled_theta[i]], [beta.pdf(sampled_theta[i], alpha[i], beta_params[i])],
                       color='red', label=f"Sampled {i}" if trial == 1 else None)
        if i == suggested_machine:
            axs[1].axvline(sampled_theta[i], color='orange', linestyle='--', label=f"Suggested Machine {i}")

    axs[1].set_title("Beta Distributions with Sampled Points")
    axs[1].set_xlabel("Probability of Success")
    axs[1].set_ylabel("Density")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def interactive_three_machines_with_visualization():
    n_machines = 3
    ts = ThompsonSampling(n_machines)
    machine_selection_counts = np.zeros(n_machines, dtype=int)
    trial = 1

    print("\n--- Thompson Sampling Interaction ---")
    print(f"There are {n_machines} machines (0, 1, 2).")
    print("At each step, you can either select a machine manually or let the system suggest one.")
    print("After selecting a machine, enter the result (1 for success, 0 for failure).")
    print("Type 'exit' to quit the simulation.\n")

    while True:
        print(f"\n--- Trial {trial} ---")
        suggested_machine, sampled_theta = ts.select_machine()
        print(f"Suggested Machine: {suggested_machine}")

        user_input = input(f"Choose a machine (0, 1, 2, or press Enter to select {suggested_machine}): ").strip()
        
        if user_input.lower() == 'exit':
            break  
        
        try:
            if user_input == "": 
                chosen_machine = suggested_machine
            else:  
                chosen_machine = int(user_input)
                if chosen_machine < 0 or chosen_machine >= n_machines:
                    raise ValueError("Invalid machine number. Choose 0, 1, or 2.")
        except ValueError:
            print("Invalid input. Please enter a valid machine number (0, 1, 2) or press Enter.")
            continue
        machine_selection_counts[chosen_machine] += 1

        while True:
            try:
                reward = int(input(f"Enter result for Machine {chosen_machine} (1 for success, 0 for failure): ").strip())
                if reward not in [0, 1]:
                    raise ValueError("Reward must be 0 or 1.")
                break
            except ValueError:
                print("Invalid input. Please enter 0 or 1.")

        ts.update(chosen_machine, reward)
        trial += 1

        print("\nCurrent machine parameters (success/failure counts):")
        for i in range(n_machines):
            print(f"Machine {i}: Alpha (success) = {ts.alpha[i]}, Beta (failure) = {ts.beta[i]}")

        plot_machine_selection_and_distributions(machine_selection_counts, trial - 1, n_machines, ts.alpha, ts.beta, sampled_theta, suggested_machine)

    print("\nSimulation ended. Final machine parameters:")
    for i in range(n_machines):
        print(f"Machine {i}: Alpha (success) = {ts.alpha[i]}, Beta (failure) = {ts.beta[i]}")

interactive_three_machines_with_visualization()
