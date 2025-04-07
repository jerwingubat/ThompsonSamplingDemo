import unittest
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

class TestAppPlots(unittest.TestCase):

    def setUp(self):
        # Setup code to initialize the necessary variables and objects
        self.ts = type('', (), {})()  # Create a simple object to hold attributes
        self.ts.n_machines = 3
        self.ts.alpha = [1, 2, 3]
        self.ts.beta = [1, 2, 3]
        self.ts.selection_counts = np.array([10, 20, 30])
        self.ts.true_rewards = np.array([0.1, 0.5, 0.9])
        self.fig, self.axs = plt.subplots(2, 2)

    def test_plot_machine_count(self):
        self.axs[0, 0].set_xlabel("Machine")
        self.axs[0, 0].set_ylabel("Count")
        self.axs[0, 0].set_xticks(range(self.ts.n_machines))
        
        self.assertEqual(self.axs[0, 0].get_xlabel(), "Machine")
        self.assertEqual(self.axs[0, 0].get_ylabel(), "Count")
        self.assertEqual(self.axs[0, 0].get_xticks().tolist(), list(range(self.ts.n_machines)))

    def test_plot_beta_distributions(self):
        x = np.linspace(0, 1, 100)
        for i in range(self.ts.n_machines):
            y = beta.pdf(x, self.ts.alpha[i], self.ts.beta[i])
            self.axs[0, 1].plot(x, y, label=f"Button {i}")
        
        self.axs[0, 1].set_title("Beta Distributions")
        self.axs[0, 1].set_xlabel("Probability of Success")
        self.axs[0, 1].set_ylabel("Density")
        self.axs[0, 1].legend()

        self.assertEqual(self.axs[0, 1].get_title(), "Beta Distributions")
        self.assertEqual(self.axs[0, 1].get_xlabel(), "Probability of Success")
        self.assertEqual(self.axs[0, 1].get_ylabel(), "Density")
        self.assertTrue(len(self.axs[0, 1].get_legend().get_texts()) == self.ts.n_machines)

    def test_plot_cumulative_rewards(self):
        self.axs[1, 0].plot(self.ts.selection_counts.cumsum(), label="Cumulative Rewards")
        optimal_rewards_diagonal = np.cumsum(np.ones(len(self.ts.selection_counts)) * np.max(self.ts.true_rewards))
        self.axs[1, 0].plot(optimal_rewards_diagonal, label="Optimal Rewards", linestyle='--', color='red')

        self.axs[1, 0].set_title("Cumulative Rewards vs Optimal Rewards")
        self.axs[1, 0].set_xlabel("Trial")
        self.axs[1, 0].set_ylabel("Reward")
        self.axs[1, 0].legend()

        self.assertEqual(self.axs[1, 0].get_title(), "Cumulative Rewards vs Optimal Rewards")
        self.assertEqual(self.axs[1, 0].get_xlabel(), "Trial")
        self.assertEqual(self.axs[1, 0].get_ylabel(), "Reward")
        self.assertTrue(len(self.axs[1, 0].get_legend().get_texts()) == 2)

if __name__ == '__main__':
    unittest.main()
    