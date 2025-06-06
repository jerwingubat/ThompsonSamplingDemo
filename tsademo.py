from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import io
import base64

app = Flask(__name__)
socketio = SocketIO(app)

class ThompsonSampling:
    def __init__(self, n_machines, true_rewards=None):
        self.n_machines = n_machines
        self.alpha = np.ones(n_machines)
        self.beta = np.ones(n_machines)
        self.selection_counts = np.zeros(n_machines, dtype=int)
        self.cumulative_rewards = 0
        self.optimal_rewards = 0
        self.total_rewards = 0

        # True success probabilities for each machine (assumed to be known for optimal policy)
        self.true_rewards = true_rewards if true_rewards is not None else np.random.uniform(0.01, 0.1, n_machines)
    
    def select_button(self):
        sampled_theta = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_theta)

    def update(self, chosen_machine, reward):
        self.selection_counts[chosen_machine] += 1
        if reward == 1:
            self.alpha[chosen_machine] += 1
        else:
            self.beta[chosen_machine] += 1

        self.cumulative_rewards += reward
        self.total_rewards += reward
        self.optimal_rewards += self.true_rewards[np.argmax(self.true_rewards)]


ts = ThompsonSampling(3)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/suggest", methods=["GET"])
def suggest():
    suggested_button = ts.select_button()
    return jsonify({"suggested_button": int(suggested_button)})

@app.route("/visualization", methods=["GET"])
def visualization():
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1
    axs[0, 0].bar(range(ts.n_machines), ts.selection_counts, color='skyblue')
    axs[0, 0].set_title("Machine Selection Counts")
    axs[0, 0].set_xlabel("Machine")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].set_xticks(range(ts.n_machines))

    # Plot 2 Beta distributions
    x = np.linspace(0, 1, 100)
    for i in range(ts.n_machines):
        y = beta.pdf(x, ts.alpha[i], ts.beta[i])
        axs[0, 1].plot(x, y, label=f"Button {i}")
    axs[0, 1].set_title("Beta Distributions")
    axs[0, 1].set_xlabel("Probability of Success")
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].legend()

    # Plot 3 Cumulative rewards vs optimal rewards
    axs[1, 0].plot(ts.selection_counts.cumsum(), label="Cumulative Rewards")
    optimal_rewards_diagonal = np.cumsum(np.ones(len(ts.selection_counts)) * np.max(ts.true_rewards))  
    axs[1, 0].plot(optimal_rewards_diagonal, label="Optimal Rewards", linestyle='--', color='red')

    axs[1, 0].set_title("Cumulative Rewards vs Optimal Rewards")
    axs[1, 0].set_xlabel("Trial")
    axs[1, 0].set_ylabel("Reward")
    axs[1, 0].legend()

    # Plot 4 Regret
    axs[1, 1].plot(np.cumsum(ts.selection_counts) - ts.cumulative_rewards, label="Regret")
    axs[1, 1].set_title("Regret")
    axs[1, 1].set_xlabel("Trial")
    axs[1, 1].set_ylabel("Regret")
    axs[1, 1].legend()

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    image_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return image_base64

@app.route("/update", methods=["POST"])
def update():
    data = request.json
    chosen_button = int(data["button"])
    reward = int(data["reward"])
    ts.update(chosen_button, reward)
    socketio.emit("update_visualization", {
        "alpha": [int(a) for a in ts.alpha],
        "beta": [int(b) for b in ts.beta],
        "selection_counts": ts.selection_counts.tolist(),
        "visualization": visualization()
    })

    return jsonify({"status": "success"})

if __name__ == "__main__":
    socketio.run(app, debug=True)
