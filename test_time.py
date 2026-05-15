import time
import numpy as np
import matplotlib.pyplot as plt

from main import solve

# ============================================================
# Configurazione
# ============================================================

STEP     = 32   # W e H devono essere multipli di STEP
MAX_W    = 256
MAX_H    = 256
REPEATS  = 1    # numero di tentativi su cui mediare il tempo

# ============================================================
# Griglia di test
# ============================================================

W_vals = list(range(STEP, MAX_W + 1, STEP))
H_vals = list(range(STEP, MAX_H + 1, STEP))

nW = len(W_vals)
nH = len(H_vals)

times  = np.full((nH, nW), np.nan)
solved = np.zeros((nH, nW), dtype=bool)

total = sum(1 for W in W_vals for H in H_vals if (W + H) % 2 == 0)
done  = 0

for j, W in enumerate(W_vals):
    for i, H in enumerate(H_vals):
        if (W + H) % 2 != 0:        # n non sarebbe multiplo di 4
            continue
        run_times = []
        any_solved = False
        for r in range(REPEATS):
            t0 = time.time()
            M, T, E = solve(W, H)
            run_times.append(time.time() - t0)
            if E == 0:
                any_solved = True

        elapsed = sum(run_times) / len(run_times)
        times[i, j]  = elapsed
        solved[i, j] = any_solved
        done += 1
        print(f"[{done:3d}/{total}] W={W:3d} H={H:3d}  "
              f"n={2*(W+H)-4:4d}  solved={solved[i,j]}  "
              f"avg={elapsed:.2f}s  runs={[f'{t:.2f}' for t in run_times]}")

# ============================================================
# Heatmap
# ============================================================

vmax = float(np.nanmax(times)) if not np.all(np.isnan(times)) else 1.0

fig, ax = plt.subplots(figsize=(max(8, nW * 1.1), max(6, nH * 0.9)))

im = ax.imshow(times, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)

for i, H in enumerate(H_vals):
    for j, W in enumerate(W_vals):
        if np.isnan(times[i, j]):
            ax.text(j, i, "–", ha="center", va="center",
                    fontsize=9, color="#888888")
            continue
        t = times[i, j]
        ok = solved[i, j]
        label = f"{t:.2f}s" + ("" if ok else "\n✗")
        fg = "white" if t > 0.6 * vmax else "black"
        ax.text(j, i, label, ha="center", va="center", fontsize=8, color=fg)

ax.set_xticks(range(nW))
ax.set_xticklabels(W_vals)
ax.set_yticks(range(nH))
ax.set_yticklabels(H_vals)
ax.set_xlabel("W", fontsize=12)
ax.set_ylabel("H", fontsize=12)
ax.set_title(f"Tempi di computazione  (STEP={STEP}, MAX_W={MAX_W}, MAX_H={MAX_H})",
             fontsize=12, pad=12)

plt.colorbar(im, ax=ax, label="Secondi", shrink=0.8)
plt.tight_layout()
plt.savefig("test_time.png", dpi=150, bbox_inches="tight")
print("Figura salvata in test_time.png")
plt.show()
