
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("catalog.csv")

# Range sulle x(offset) per i plot dei modelli
x = np.linspace(0, 1.5, 400)

# DEFINZIONE DEI MODELLI

#  MODELLO S15 
def s15_pdf(x, rho, sigma0, sigma1):
    """
    S15: mixture di due Rayleigh
    """
    term1 = rho * (x / sigma0**2) * np.exp(-0.5 * (x / sigma0)**2)
    term2 = (1 - rho) * (x / sigma1**2) * np.exp(-0.5 * (x / sigma1)**2)
    return term1 + term2

# MODELLO Z19
def z19_pdf(x, rho, sigma, tau):
    """
    Z19: mixture di due componenti sempre positive
    P(x) = rho * (1/sigma) * exp(-x/sigma) + (1-rho) * (x/tau^2) * exp(-x/tau)
    """
    term1 = rho * (1 / sigma) * np.exp(-x / sigma)
    term2 = (1 - rho) * (x / tau**2) * np.exp(-x / tau)
    return term1 + term2


# PARAMETRI DI RIFERIMENTO
rho0 = 0.7
sigma0_0 = 0.05   # per S15
sigma1_0 = 0.3    # per S15
sigma_z19 = 0.05  # componente centrata Z19
tau_z19 = 0.3     # componente mis-centrata Z19


# ANALISI DELLA VARIAZIONE DEI PARAMETRI


# Variazione di rho
rhos = [0.2, 0.5, 0.8]

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
for r in rhos:
    plt.plot(x, s15_pdf(x, r, sigma0_0, sigma1_0), label=f"rho={r}")
plt.title("S15: varying rho")
plt.xlabel("x")
plt.ylabel("PDF")
plt.ylim(0, None)
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1,2,2)
for r in rhos:
    plt.plot(x, z19_pdf(x, r, sigma_z19, tau_z19), label=f"rho={r}")
plt.title("Z19: varying rho")
plt.xlabel("x")
plt.ylim(0, None)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()


# Variazione della componente centrata (sigma0 per S15, sigma per Z19)

sigma0_values = [0.02, 0.05, 0.1]

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
for s0 in sigma0_values:
    plt.plot(x, s15_pdf(x, rho0, s0, sigma1_0), label=f"sigma0={s0}")
plt.title("S15: varying sigma0")
plt.xlabel("x")
plt.ylabel("PDF")
plt.ylim(0, None)
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1,2,2)
for s in sigma0_values:
    plt.plot(x, z19_pdf(x, rho0, s, tau_z19), label=f"sigma={s}")
plt.title("Z19: varying sigma (centrata)")
plt.xlabel("x")
plt.ylim(0, None)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Variazione della componente mis-centrata (sigma1 per S15, tau per Z19)
sigma1_values = [0.1, 0.3, 0.6]

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
for s1 in sigma1_values:
    plt.plot(x, s15_pdf(x, rho0, sigma0_0, s1), label=f"sigma1={s1}")
plt.title("S15: varying sigma1")
plt.xlabel("x")
plt.ylabel("PDF")
plt.ylim(0, None)
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1,2,2)
for t in sigma1_values:
    plt.plot(x, z19_pdf(x, rho0, sigma_z19, t), label=f"tau={t}")
plt.title("Z19: varying tau (mis-centrata)")
plt.xlabel("x")
plt.ylim(0, None)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
