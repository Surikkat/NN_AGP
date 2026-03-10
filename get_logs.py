import os
import numpy as np

BASE_DIR = "results"
STEPS = [100, 200]


def get_value(arr, step):
    idx = step - 1
    return float(arr[idx])


print("\n===== Кумулятивный регрет =====\n")
print(f"{'Exp':<6} {'Algorithm':<22} {'Reg@100':<12} {'Reg@200':<12}")
print("-" * 60)

for exp_id in range(1, 11):
    exp_path = os.path.join(BASE_DIR, f"exp_{exp_id}", "npz_results")

    if not os.path.isdir(exp_path):
        continue

    npz_files = [f for f in os.listdir(exp_path) if f.endswith(".npz")]
    if not npz_files:
        continue

    for file in npz_files:
        full_path = os.path.join(exp_path, file)
        data = np.load(full_path, allow_pickle=True)

        # Фиксированные m
        if "regrets_fixed" in data.files:
            fixed_dict = data["regrets_fixed"].item()
            for m, curve in fixed_dict.items():
                r100 = get_value(curve, 100)
                r200 = get_value(curve, 200)
                print(f"{exp_id:<6} NN-AGP (m={m}){'':<8} {r100:<12.2f} {r200:<12.2f}")

        # Остальные алгоритмы
        mapping = {
            "regrets_adapt": "NN-AGP Adaptive",
            "regrets_gp_mult": "GP-UCB Mult",
            "regrets_gp_adapt": "GP-UCB Adapt",
            "regrets_neuralucb": "NeuralUCB",
            "regrets_nnucb": "NN-UCB",
        }

        for key, name in mapping.items():
            if key in data.files:
                curve = data[key]
                r100 = get_value(curve, 100)
                r200 = get_value(curve, 200)
                print(f"{exp_id:<6} {name:<22} {r100:<12.2f} {r200:<12.2f}")

print("\n=====================================\n")

