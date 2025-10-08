import matplotlib.pyplot as plt
import pandas as pd
import os

df = pd.read_csv("predicted_Vout.csv")

sim0 = df[df["sim_id"] == 0].sort_values("t_s")

plt.figure(figsize=(8, 4))
plt.plot(sim0["t_s"], sim0["Vout"], marker='o')
plt.title("Waveform sim_id = 0")
plt.xlabel("t_s")
plt.ylabel("Vout")
plt.grid(True)
plt.legend()

outpath = os.path.join("Wave0.png")
plt.savefig(outpath)
plt.close()