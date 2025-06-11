import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

# ------------------ SETTINGS --------------------#
W_max_fit = 2.0          # Only fit data with W <= this value
Q2_bins_to_fit = 9       # Number of lowest Q² bins to include in fit (data will be fittted up until this bin)
E0 = 10.6                # Beam energy in GeV
M = 0.938                # Proton mass in GeV
alpha = 1/137            # Fine structure constant
R_fixed = 0.18           # R parametrization as in the Breibenbach paper
# ------------------------------------------------#

# Load data
df = pd.read_csv("exp_data_all.dat")

# Compute derived quantities
df["nu"] = (df["W"]**2 + df["Q2"] - M**2) / (2 * M)
df["omega_prime"] = 1 + (df["W"]**2 / df["Q2"])   # for the scaling function as in the paper
df["E_prime"] = E0 - df["nu"]
df["cos_theta"] = 1 - df["Q2"] / (2 * E0 * df["E_prime"])
df["theta_rad"] = np.arccos(df["cos_theta"].clip(-1, 1))
df["uncertainty"] = np.sqrt(df["Stat"]**2 + df["Sys"]**2)
df["x"] = df["Q2"] / (2 * M * df["nu"])

# Q² bins
q2_bins_all = sorted(df["Q2"].unique())
q2_bins_fit = q2_bins_all[:Q2_bins_to_fit]                       # maybe we'd want to fit only some Q2, not all of them?
df_fit = df[df["Q2"].isin(q2_bins_fit) & (df["W"] <= W_max_fit)]

# Resonance parameters (fixed masses and widths as in the paper) Maybe add more resonances? 
resonance_params = [
    {"M": 1.226, "Gamma": 0.115},
    {"M": 1.508, "Gamma": 0.080},
    {"M": 1.705, "Gamma": 0.085},
    {"M": 1.920, "Gamma": 0.220},
]

# -------------------- Threshold-corrected resonance R₁(W) - described in th paper on second page under ++ appendix ----------------

R_iso = 0.8  # isobar radius in fm
m_pi = 0.13957  # pion mass in GeV
M = 0.938  # nucleon mass in GeV (already defined)
M1 = 1.226  # mass of first resonance (GeV)
Gamma1 = 0.115  # width of first resonance (GeV)

def q_star(W, M, m_pi):
    num = (W**2 - (M + m_pi)**2) * (W**2 - (M - m_pi)**2)
    return np.sqrt(np.clip(num, 0, None)) / (2 * W)

q_star_0 = q_star(M1, M, m_pi)  # at resonance

def V(q, R):
    return q**2 / (1 + q**2 * R**2)

def Gamma_R(W, Gamma1, R=R_iso):
    q = q_star(W, M, m_pi)
    return Gamma1 * (V(q, R) / V(q_star_0, R)) * (q / q_star_0)

def R1_modified(W):
    W = np.atleast_1d(W)
    R1 = np.zeros_like(W)
    below = W < M1
    above = ~below

    q = q_star(W[below], M, m_pi)
    gamma_r = Gamma_R(W[below], Gamma1)
    BW = (gamma_r**2 * M1**2) / ((W[below]**2 - M1**2)**2 + gamma_r**2 * M1**2)
    pre = (M / W[below]) * ((W[below]**2 - M**2) / (M1**2 - M**2))**2 * (q / q_star_0)**3
    R1[below] = pre * BW

    gamma1_sq = Gamma1**2
    R1[above] = (gamma1_sq * M1**2) / ((W[above]**2 - M1**2)**2 + gamma1_sq * M1**2)

    return R1 if len(R1) > 1 else R1[0]
# ------------------------------------------------------------------------------------------------------------------------#

# Model function
def model(W, Q2, nu, theta, omega_p, x, a1, a2, a3, a4, b1, b2, b3, c1, c2, c3):
# Use threshold-modified R₁
    R1 = a1 * R1_modified(W)

    # Standard Breit-Wigner for R₂–R₄
    R_rest = sum([
        a * (Gamma**2 * Mres**2) / ((W**2 - Mres**2)**2 + Gamma**2 * Mres**2)
        for a, r in zip([a2, a3, a4], resonance_params[1:])
        for Mres, Gamma in [(r["M"], r["Gamma"])]
    ])

    R = R1 + R_rest # resonance function as in the paper (formula (2))

    Wt = 1.08
    B = 1 - (b1 / (1 + (W - Wt)**2)) - (b2 / (1 + (W - Wt)**2)**2) - (b3 / (1 + (W - Wt)**2)**3)  # background function as in the paper (formula (2)). Maybe add interference of bg and resonance?
    
    F2 = ( c1 * (1 - 1 / omega_p)**3 + c2 * (1 - 1 / omega_p)**4 + c3 * (1 - 1 / omega_p)**5 ) # scaling function as in the paper (formula (3)). Maybe add another power 6?
    
    
    W2 = F2 * (R + B)/nu    # calculating structure functions as in the paper (formula (1))
    W1 = W2 / (2 * x * (1 + R_fixed)) # known formula. Follows fron Callan-Gross relation for dimentionless F1,F2 structure functions
    
    # calculating differential cross section as in the pdf write-up
    dsigma = ((alpha**2  * np.cos(theta / 2)**2) / (4 * E0**2 * np.sin(theta / 2)**4)) * (E0 - nu) / E0 * (W2 + 2 * np.tan(theta / 2)**2 * W1)
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
           a1=0.7, a2=0.4, a3=0.3, a4=0.1,  # initial guesses for resonance parameters
           b1=0.5, b2=0.5, b3=0.5,          # initial guesses for background parameters
           c1=1.0, c2=1.0, c3=1.0)          # initial guesses for scaling function parameters
m.migrad()


# Save fit results to .txt file
with open(f"fit_results_W_max={W_max_fit}_Q2_bins_fitted={Q2_bins_to_fit}.txt", "w") as f_out:
    f_out.write(f"Fit results for W_max_fit = {W_max_fit}, Q² bins = {Q2_bins_to_fit}\n")
    f_out.write("=" * 50 + "\n")
    f_out.write("Minimization status:\n")
    f_out.write(f"  Valid minimization: {m.fmin.is_valid}\n")
    f_out.write(f"  Converged (EDM < tol): {m.fmin.edm < m.tol} (EDM = {m.fmin.edm:.2e}, tol = {m.tol})\n")
    f_out.write("\nParameter values:\n")
    for name in m.parameters:
        val = m.values[name]
        err = m.errors[name]
        f_out.write(f"  {name:>4} = {val:.6f} ± {err:.6f}\n")
    f_out.write("=" * 50 + "\n")
    f_out.write(f"Chi2 / NDF = {m.fval:.2f} / {m.ndof} = {m.fval / m.ndof:.3f}\n")

    # Write full symmetric correlation matrix
    param_names = m.parameters
    f_out.write("\nFull correlation matrix:\n\n")

    # Header
    f_out.write(f"{'':>8}" + "".join(f"{name:>10}" for name in param_names) + "\n")

    # Matrix rows
    for name1 in param_names:
        f_out.write(f"{name1:>8}")
        for name2 in param_names:
            val = m.covariance[name1, name2]
            f_out.write(f"{val:10.3f}")
        f_out.write("\n")




# Plotting - nothing interesting here
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
        ax.set_ylabel(r"$d^2\sigma/dWdQ^2$ [$mb/GeV^3$]")
    if idx >= 6:
        ax.set_xlabel("W [GeV]")
    ax.legend(fontsize=7)

# Finalize and save plot
fig.suptitle("Fit to Inclusive Cross Section Data", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"fit_data_W_max={W_max_fit}_Q2_bins_fitted={Q2_bins_to_fit}.png", dpi=300)
plt.show()
