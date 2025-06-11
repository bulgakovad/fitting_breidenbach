import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

# ------------------ SETTINGS ------------------
W_max_fit = 2.0          # Only fit data with W <= this value
Q2_bins_to_fit = 9       # Number of lowest Q² bins to include in fit
E0 = 10.6                # Beam energy in GeV
M = 0.938                # Proton mass in GeV
alpha = 1/137
R_fixed = 0.18
# ------------------------------------------------

# Load data
df = pd.read_csv("exp_data_all.dat")

# Compute derived quantities
df["nu"] = (df["W"]**2 + df["Q2"] - M**2) / (2 * M)
df["omega_prime"] = 1 + (df["W"]**2 / df["Q2"])
df["E_prime"] = E0 - df["nu"]
df["cos_theta"] = 1 - df["Q2"] / (2 * E0 * df["E_prime"])
df["theta_rad"] = np.arccos(df["cos_theta"].clip(-1, 1))
df["mott"] = (
    (alpha**2 * np.cos(df["theta_rad"] / 2)**2) /
    (4 * E0**2 * np.sin(df["theta_rad"] / 2)**4)
) * (df["E_prime"] / E0)
df["uncertainty"] = np.sqrt(df["Stat"]**2 + df["Sys"]**2)
df["jacobian"] = np.pi * df["W"] / (M * E0 * df["E_prime"])
df["x"] = df["Q2"] / (2 * M * df["nu"])

# Q² bins
q2_bins_all = sorted(df["Q2"].unique())
q2_bins_fit = q2_bins_all[:Q2_bins_to_fit]
df_fit = df[df["Q2"].isin(q2_bins_fit) & (df["W"] <= W_max_fit)]

# Resonance parameters (fixed masses and widths)
resonance_params = [
    {"M": 1.226, "Gamma": 0.115},
    {"M": 1.508, "Gamma": 0.080},
    {"M": 1.705, "Gamma": 0.085},
    {"M": 1.920, "Gamma": 0.220},
]

# Model function
def model(W, Q2, nu, theta, omega_p, x, a1, a2, a3, a4, b1, b2, b3, c1, c2, c3):
    R = sum([
        a * (Gamma**2 * Mres**2) / ((W**2 - Mres**2)**2 + Gamma**2 * Mres**2)
        for a, r in zip([a1, a2, a3, a4], resonance_params)
        for Mres, Gamma in [(r["M"], r["Gamma"])]
    ])
    Wt = 1.08
    B = 1 - (b1 / (1 + (W - Wt)**2)) - (b2 / (1 + (W - Wt)**2)**2) - (b3 / (1 + (W - Wt)**2)**3)
    F2 = (
        c1 * (1 - 1 / omega_p)**3 +
        c2 * (1 - 1 / omega_p)**4 +
        c3 * (1 - 1 / omega_p)**5
    )
    W2 = F2 * (R + B)
    W1 = W2 / (2 * x * (1 + R_fixed))
    dsigma = (
        (alpha**2 * np.cos(theta / 2)**2) / (4 * E0**2 * np.sin(theta / 2)**4)
    ) * (E0 - nu) / E0 * (W2 + 2 * np.tan(theta / 2)**2 * W1)
    return dsigma * (np.pi * W / (M * E0 * (E0 - nu)))

# Wrap model for LeastSquares
def wrapped_model(x_tuple, a1, a2, a3, a4, b1, b2, b3, c1, c2, c3):
    W, Q2, nu, theta, omega_p, x = x_tuple
    return model(W, Q2, nu, theta, omega_p, x, a1, a2, a3, a4, b1, b2, b3, c1, c2, c3)

# Prepare data for fit
xdata = tuple(col.to_numpy() for col in [
    df_fit["W"], df_fit["Q2"], df_fit["nu"], df_fit["theta_rad"],
    df_fit["omega_prime"], df_fit["x"]
])
ydata = df_fit["XSEC"].to_numpy()
yerr = df_fit["uncertainty"].to_numpy()

# Fit with Minuit
least_squares = LeastSquares(xdata, ydata, yerr, wrapped_model)
m = Minuit(least_squares,
           a1=0.7, a2=0.4, a3=0.3, a4=0.1,
           b1=0.5, b2=0.5, b3=0.5,
           c1=1.0, c2=1.0, c3=1.0)
m.migrad()

# Save fit results to TXT
with open(f"fit_results_W_max={W_max_fit}.txt", "w") as f_out:
    f_out.write(f"Fit results for W_max_fit = {W_max_fit}, Q² bins = {Q2_bins_to_fit}\n")
    f_out.write("=" * 40 + "\n")
    for name in m.parameters:
        val = m.values[name]
        err = m.errors[name]
        f_out.write(f"{name:>4} = {val:.6f} ± {err:.6f}\n")
    f_out.write("=" * 40 + "\n")
    f_out.write(f"Chi2 / NDF = {m.fval:.2f} / {m.ndof} = {m.fval / m.ndof:.3f}\n")

# Plotting
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
w_vals = np.linspace(df["W"].min(), df["W"].max(), 300)

for idx, q2 in enumerate(q2_bins_all):
    ax = axes[idx]
    subdf = df[df["Q2"] == q2]
    ax.errorbar(
        subdf["W"].to_numpy(),
        subdf["XSEC"].to_numpy(),
        yerr=subdf["uncertainty"].to_numpy(),
        fmt='o',
        markersize=3,
        label=f"Q²={q2:.2f}"
    )

    if q2 in q2_bins_fit:
        mask_fit = w_vals <= W_max_fit
        w_plot = w_vals[mask_fit]
        nu_vals = (w_plot**2 + q2 - M**2) / (2 * M)
        omega_vals = 1 + w_plot**2 / q2
        E_prime_vals = E0 - nu_vals
        theta_vals = np.arccos((1 - q2 / (2 * E0 * E_prime_vals)).clip(-1, 1))
        x_vals = q2 / (2 * M * nu_vals)

        fit_vals = model(w_plot, q2, nu_vals, theta_vals, omega_vals, x_vals, *m.values)
        ax.plot(w_plot, fit_vals, '-', color='black', linewidth=1.3, label="Full Fit")

        R_vals = model(w_plot, q2, nu_vals, theta_vals, omega_vals, x_vals,
                       m.values["a1"], m.values["a2"], m.values["a3"], m.values["a4"],
                       0, 0, 0, m.values["c1"], m.values["c2"], m.values["c3"])
        ax.plot(w_plot, R_vals, '--', color='blue', label="Resonance")

        B_vals = model(w_plot, q2, nu_vals, theta_vals, omega_vals, x_vals,
                       0, 0, 0, 0,
                       m.values["b1"], m.values["b2"], m.values["b3"],
                       m.values["c1"], m.values["c2"], m.values["c3"])
        ax.plot(w_plot, B_vals, '--', color='red', label="Background")

    ax.axvline(W_max_fit, linestyle='--', color='gray', linewidth=1)
    ax.set_title(f"Q² = {q2:.2f} GeV²")
    ax.grid(True)
    if idx % 3 == 0:
        ax.set_ylabel(r"$d^2\sigma/dWdQ^2$ [mb/GeV²]")
    if idx >= 6:
        ax.set_xlabel("W [GeV]")
    ax.legend(fontsize=7)

# Finalize and save plot
fig.suptitle("Fit to Inclusive Cross Section Data", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"fit_data_W_max={W_max_fit}.png", dpi=300)
plt.show()
