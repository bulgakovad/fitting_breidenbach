import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

# ------------------ SETTINGS --------------------#
W_max_fit = 2.0         # Fit data with W <= this value
E0 = 10.6               # Beam energy in GeV
M = 0.938               # Proton mass in GeV
alpha = 1/137           # Fine structure constant
R_fixed = 0.18          # R parametrization
# ------------------------------------------------#

# Load new pseudodata
df = pd.read_csv("pseudodata.dat", delim_whitespace=True, comment='#',
                 names=["Q2", "W", "XSEC", "error"])

# Derived quantities
df["nu"] = (df["W"]**2 + df["Q2"] - M**2) / (2 * M)
df["omega_prime"] = 1 + (df["W"]**2 / df["Q2"])
df["E_prime"] = E0 - df["nu"]
df["cos_theta"] = 1 - df["Q2"] / (2 * E0 * df["E_prime"])
df["theta_rad"] = np.arccos(df["cos_theta"].clip(-1, 1))
df["x"] = df["Q2"] / (2 * M * df["nu"])

# Q² bins
q2_bins_all = sorted(df["Q2"].unique())
df_fit = df[df["W"] <= W_max_fit]

# ------------------- Model without resonance ------------------- #
def model_no_resonance(W, Q2, nu, theta, omega_p, x, b1, b2, b3, c1, c2, c3):
    # No resonance term
    R = 0.0
    Wt = 1.08
    B = 1 - (b1 / (1 + (W - Wt)**2)) - (b2 / (1 + (W - Wt)**2)**2) - (b3 / (1 + (W - Wt)**2)**3)
    F2 = ( c1 * (1 - 1 / omega_p)**3 + c2 * (1 - 1 / omega_p)**4 + c3 * (1 - 1 / omega_p)**5 )
    W2 = F2 * (R + B) / nu
    W1 = W2 / (2 * x * (1 + R_fixed))
    dsigma = ((alpha**2 * np.cos(theta / 2)**2) / (4 * E0**2 * np.sin(theta / 2)**4)) * (E0 - nu) / E0 * (W2 + 2 * np.tan(theta / 2)**2 * W1)
    return dsigma * (np.pi * W / (M * E0 * (E0 - nu)))

# Wrapper for iminuit
def wrapped_model(x_tuple, b1, b2, b3, c1, c2, c3):
    W, Q2, nu, theta, omega_p, x = x_tuple
    return model_no_resonance(W, Q2, nu, theta, omega_p, x, b1, b2, b3, c1, c2, c3)

# Prepare data
xdata = tuple(col.to_numpy() for col in [
    df_fit["W"], df_fit["Q2"], df_fit["nu"], df_fit["theta_rad"],
    df_fit["omega_prime"], df_fit["x"]
])
ydata = df_fit["XSEC"].to_numpy()
yerr = df_fit["error"].to_numpy()

# Fit with Minuit
least_squares = LeastSquares(xdata, ydata, yerr, wrapped_model)
m = Minuit(least_squares, b1=0.5, b2=0.5, b3=0.5, c1=1.0, c2=1.0, c3=1.0)
m.migrad()

# ------------------- Save Results ------------------- #
out_txt = "fit_results_pseudodata_no_resonance.txt"
with open(out_txt, "w") as f_out:
    f_out.write("Fit results (no resonance, pseudodata)\n")
    f_out.write("=" * 50 + "\n")
    for name in m.parameters:
        f_out.write(f"{name:>4} = {m.values[name]:.6f} ± {m.errors[name]:.6f}\n")
    f_out.write("=" * 50 + "\n")
    f_out.write(f"Chi2 / NDF = {m.fval:.2f} / {m.ndof} = {m.fval / m.ndof:.3f}\n")

# ------------------- Plot ------------------- #
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
w_vals = np.linspace(df["W"].min(), df["W"].max(), 300)

for idx, q2 in enumerate(q2_bins_all[:9]):  # max 9 subplots
    ax = axes[idx]
    subdf = df[df["Q2"] == q2]
    ax.errorbar(
        subdf["W"].to_numpy(),
        subdf["XSEC"].to_numpy(),
        yerr=subdf["error"].to_numpy(),
        fmt='o', markersize=3,
        label=f"Q²={q2:.2f}"
    )
    mask_fit = w_vals <= W_max_fit
    w_plot = w_vals[mask_fit]
    nu_vals = (w_plot**2 + q2 - M**2) / (2 * M)
    omega_vals = 1 + w_plot**2 / q2
    E_prime_vals = E0 - nu_vals
    theta_vals = np.arccos((1 - q2 / (2 * E0 * E_prime_vals)).clip(-1, 1))
    x_vals = q2 / (2 * M * nu_vals)

    fit_vals = model_no_resonance(w_plot, q2, nu_vals, theta_vals, omega_vals, x_vals, *m.values)
    ax.plot(w_plot, fit_vals, '-', color='black', linewidth=1.3, label="Fit")

    ax.axvline(W_max_fit, linestyle='--', color='gray', linewidth=1)
    ax.set_title(f"Q² = {q2:.2f} GeV²")
    ax.grid(True)
    if idx % 3 == 0:
        ax.set_ylabel(r"$d^2\sigma/dWdQ^2$ [$mb/GeV^3$]")
    if idx >= 6:
        ax.set_xlabel("W [GeV]")
    ax.legend(fontsize=7)

fig.suptitle("Fit to Pseudodata (No Resonance)", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("fit_pseudodata_no_resonance.png", dpi=300)
plt.show()
