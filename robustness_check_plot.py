
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


if __name__ == "__main__":
    
    ### Embedding generation
    # 1) Walk bias
    x = np.array([0, 0.5])
    width = 0.1
    x_values = ["Ratio 1:2:3", "Ratio 1:3:5"]
    y_sqrt_values = [0.643, 0.734]
    y_log_values = [0.646, 0.688]
    y_orig_values = [0.573, 0.601]


    plt.figure(figsize=(6,4))
    plt.bar(x - width, y_sqrt_values, width, label="Sqrt")
    plt.bar(x, y_log_values, width, label="Log", color='sandybrown')
    plt.bar(x + width , y_orig_values, width, label="Raw", color='forestgreen')
    plt.xlabel("Relative probability")
    plt.ylabel("AUC")
    plt.xticks(x, x_values)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/walk_bias_robustness_check.png")
    plt.close()

    # 2) Embedding size
    x_values = ["8", "16", "32", "64"]
    y_values = [0.674, 0.57, 0.734, 0.668]
    
    plt.figure(figsize=(6,4))
    plt.plot(x_values, y_values, marker='o', label='Node embedding size')
    plt.xlabel("Node embedding size ($d_{collab}$)")
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.savefig("plots/embedding_size_robustness_check.png")
    plt.close()


    ### Prediction
    # 1) learning rate
    x_values = ["$10^{-1}$", "$10^{-2}$", "$10^{-3}$", "$10^{-4}$"]
    y_values = [0.582, 0.708, 0.734, 0.719]

    plt.figure(figsize=(6,4))
    plt.plot(x_values, y_values, marker='o')
    plt.xlabel("Learning rate")
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.savefig("plots/learing_rate_robustness_check.png")
    plt.close()


    # 2) Dropout rate
    x_values = ["0.2", "0.4", "0.6", "0.8"]
    y_values = [0.709, 0.734, 0.727, 0.713]

    plt.figure(figsize=(6,4))
    plt.plot(x_values, y_values, marker='o')
    plt.xlabel("Dropout rate")
    plt.ylabel("AUC")
    plt.savefig("plots/dropout_rate_robustness_check.png")
    plt.close()