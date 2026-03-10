import numpy as np
import matplotlib.pyplot as plt

def load_results(path):
    return np.load(path, allow_pickle=True)


path = input("Введите путь для результатов .npz: ")
data = load_results(path)

regrets_fixed = data["regrets_fixed"].item()
regrets_adapt = data["regrets_adapt"]
regrets_gp_mult = data["regrets_gp_mult"]
regrets_gp_adapt = data["regrets_gp_adapt"]
regrets_neuralucb = data["regrets_neuralucb"]
regrets_nnucb = data["regrets_nnucb"]

inst_fixed = data["inst_fixed"].item()
inst_adapt = data["inst_adapt"]
inst_gp_mult = data["inst_gp_mult"]
inst_gp_adapt = data["inst_gp_adapt"]
inst_neuralucb = data["inst_neuralucb"]
inst_nnucb = data["inst_nnucb"]

m_history = data["m_history"]
T = int(data["T"])
t_axis = np.arange(1, T + 1)

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16
})

plt.figure(figsize=(14, 6), dpi=350)

for m in regrets_fixed:
    avg_r = np.array(inst_fixed[m]).cumsum() / t_axis
    plt.plot(t_axis, avg_r, label=f"Fixed m={m}", linewidth=2)

plt.plot(t_axis, np.array(inst_adapt).cumsum() / t_axis, label="Adaptive NN-AGP", linewidth=3)
plt.plot(t_axis, np.array(inst_gp_mult).cumsum() / t_axis, label="GP-UCB Multiplicative", linewidth=3)
plt.plot(t_axis, np.array(inst_gp_adapt).cumsum() / t_axis, label="GP-UCB Adaptive", linewidth=3)
plt.plot(t_axis, np.array(inst_neuralucb).cumsum() / t_axis, label="NeuralUCB", linewidth=3)
plt.plot(t_axis, np.array(inst_nnucb).cumsum() / t_axis, label="NN-UCB", linewidth=3)
plt.xlabel("Rounds")
plt.ylabel("Average regret")
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=4,                 
    frameon=False,          
    fontsize=15,            
    columnspacing=1.5,      
    handlelength=2.5        
)
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
plt.savefig("Figure_1.png", dpi=350, bbox_inches="tight")
#plt.show()


plt.figure(figsize=(14, 6), dpi=350)

for m in regrets_fixed:
    plt.plot(t_axis, regrets_fixed[m], label=f"Fixed m={m}", linewidth=2)

plt.plot(t_axis, regrets_adapt, label="Adaptive NN-AGP", linewidth=3)
plt.plot(t_axis, regrets_gp_mult, label="GP-UCB Multiplicative", linewidth=3)
plt.plot(t_axis, regrets_gp_adapt, label="GP-UCB Adaptive", linewidth=3)
plt.plot(t_axis, regrets_neuralucb, label="NeuralUCB", linewidth=3)
plt.plot(t_axis, regrets_nnucb, label="NN-UCB", linewidth=3)
plt.xlabel("Rounds")
plt.ylabel("Cumulative regret")
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=4,                 
    frameon=False,          
    fontsize=15,            
    columnspacing=1.5,      
    handlelength=2.5        
)
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
plt.savefig("Figure_2.png", dpi=350, bbox_inches="tight")
#plt.show()

plt.figure(figsize=(8, 3), dpi=350)
plt.plot(t_axis, m_history, marker='o', markersize=3, linewidth=2)
plt.xlabel("Rounds")
plt.ylabel("Effective m")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Figure_3.png", dpi=350, bbox_inches="tight")
#plt.show()