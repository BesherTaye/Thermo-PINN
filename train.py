import torch
import matplotlib.pyplot as plt
import pandas as pd
from model import PINN_Trainer

n_int = 256
n_sb = 64
n_tb = 64

pinn = PINN_Trainer(n_int, n_sb, n_tb)

# Run the training and collect loss history
history_total, history_pde, history_function = pinn.fit(num_epochs=1, verbose=True)


# Plot losses separately
plt.figure(figsize=(12, 6), dpi=150)

plt.subplot(1, 3, 1)
plt.plot(history_total, label="Total Loss", color='blue')
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
plt.title("Total Loss Evolution")
plt.grid(True, linestyle=":")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history_pde, label="PDE Loss", color='orange')
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
plt.title("PDE Loss Evolution")
plt.grid(True, linestyle=":")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history_function, label="Function Loss", color='red')
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
plt.title("Function Loss Evolution")
plt.grid(True, linestyle=":")
plt.legend()

plt.tight_layout()

plt.savefig("loss_evolution.png", dpi=300, bbox_inches="tight")

pinn.plot()
plt.savefig("approx_solution.png", dpi=300, bbox_inches="tight")