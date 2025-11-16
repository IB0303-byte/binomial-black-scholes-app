# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Binomial vs Black-Scholes + 3D + Delta Hedge", page_icon="📈", layout="wide")

st.title("📊 Convergence binomial → Black–Scholes + Simulateur 3D et Couverture Δ")

st.markdown(
    "Cette application montre la convergence du modèle binomial vers Black–Scholes, "
    "ajoute une surface 3D du prix binomial (en fonction de S₀ et N) et simule une "
    "couverture Δ (hedging) réalisée via rééquilibrage à chaque étape binomiale."
)

# -----------------------
# Inputs
# -----------------------
with st.sidebar:
    st.header("Paramètres généraux")
    S0 = st.number_input("Prix initial S₀", value=100.0, min_value=0.01)
    K = st.number_input("Strike K", value=100.0, min_value=0.0)
    r = st.number_input("Taux sans risque r", value=0.05, step=0.01, min_value=0.0)
    sigma = st.number_input("Volatilité σ", value=0.2, step=0.01, min_value=0.0)
    T = st.number_input("Maturité T (années)", value=1.0, step=0.01, min_value=0.01)
    option_type = st.selectbox("Type d'option", ["Call", "Put"])
    max_N = st.slider("Nombre maximal d'étapes (N_max) pour 3D", 10, 300, 100)
    n_S_grid = st.slider("Résolution surface (S grid size)", 10, 60, 30)
    run_hedge = st.checkbox("Simuler couverture delta (Monte Carlo)", value=True)
    if run_hedge:
        n_sims = st.number_input("Nombre de simulations MC", value=200, min_value=10, max_value=2000, step=10)
        rebalancing_N = st.slider("Rééquilibrage (N étapes binomiales pour hedge)", 1, 200, 50)

# -----------------------
# Pricing functions
# -----------------------
def black_scholes(S0, K, r, sigma, T, option_type="Call"):
    if S0 <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

def binomial_option_price(S0, K, r, sigma, T, N, option_type="Call"):
    # N must be >=1
    N = int(max(1, N))
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    q = (np.exp(r * dt) - d) / (u - d)
    # terminal stock prices
    j = np.arange(N, -1, -1)
    ST = S0 * (d ** j) * (u ** (N - j))
    if option_type == "Call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    # backward induction
    for i in range(N, 0, -1):
        payoff = disc * (q * payoff[1:] + (1 - q) * payoff[:-1])
    return float(payoff[0])

def binomial_delta(S0, K, r, sigma, T, N, option_type="Call"):
    """
    Compute binomial delta at current time using one-step relation:
    Δ = (C_up - C_down) / (S0*u - S0*d)
    where C_up/C_down are option values at the next nodes (computed via binomial with N-1 steps).
    """
    N = int(max(1, N))
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    # next-node option values:
    if N == 1:
        # terminal payoffs at next step
        Su = S0 * u
        Sd = S0 * d
        if option_type == "Call":
            Cu = max(Su - K, 0.0)
            Cd = max(Sd - K, 0.0)
        else:
            Cu = max(K - Su, 0.0)
            Cd = max(K - Sd, 0.0)
    else:
        # option value at next nodes: treat them as roots for N-1 steps
        Cu = binomial_option_price(S0 * u, K, r, sigma, T - dt, N - 1, option_type)
        Cd = binomial_option_price(S0 * d, K, r, sigma, T - dt, N - 1, option_type)
    denom = S0 * (u - d)
    if denom == 0:
        return 0.0
    return float((Cu - Cd) / denom)

# -----------------------
# Main visual: convergence plot
# -----------------------
st.subheader("🔢 Convergence : prix binomial vs Black–Scholes")
col1, col2 = st.columns([2, 1])

with col1:
    default_N = min(max_N, 100)
    steps = st.slider("Choisir N (pour affichage comparatif)", 1, max_N, default_N)
    bs_price = black_scholes(S0, K, r, sigma, T, option_type)
    # compute binomial prices for a selection of N
    Ns = np.unique(np.linspace(1, max_N, min(80, max_N), dtype=int))
    bin_prices = [binomial_option_price(S0, K, r, sigma, T, n, option_type) for n in Ns]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Ns, y=bin_prices, mode="lines+markers", name="Prix Binomial"))
    fig.add_trace(go.Scatter(x=Ns, y=[bs_price] * len(Ns), mode="lines", name="Black–Scholes", line=dict(dash="dash")))
    fig.update_layout(title="Convergence du prix binomial vers Black–Scholes",
                      xaxis_title="Nombre d'étapes N", yaxis_title="Prix de l'option",
                      template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Prix Black–Scholes", f"{bs_price:.6f}")
    st.metric(f"Prix Binomial (N = {steps})", f"{binomial_option_price(S0, K, r, sigma, T, steps, option_type):.6f}")
    st.markdown("**Δ (binomial)** à la racine : "
                f"{binomial_delta(S0, K, r, sigma, T, steps, option_type):.6f}")

# -----------------------
# 3D Surface: prix en fonction de S0 et N
# -----------------------
st.subheader("🟨 Simulateur 3D — Prix binomial en fonction de S₀ et N")

col3, col4 = st.columns([2, 1])
with col3:
    S_min = max(0.01, 0.5 * S0)
    S_max = 1.5 * S0
    S_vals = np.linspace(S_min, S_max, n_S_grid)
    N_vals = np.unique(np.linspace(1, max_N, min(60, max_N), dtype=int))

    # compute surface
    Z = np.zeros((len(S_vals), len(N_vals)))
    for i, Sval in enumerate(S_vals):
        for j, Nj in enumerate(N_vals):
            Z[i, j] = binomial_option_price(Sval, K, r, sigma, T, Nj, option_type)

    # build surface plot
    X, Y = np.meshgrid(N_vals, S_vals)
    surf = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    surf.update_layout(title="Surface : Prix binomial(S₀, N)",
                       scene=dict(xaxis_title="N (étapes)", yaxis_title="S₀", zaxis_title="Prix"),
                       height=600)
    st.plotly_chart(surf, use_container_width=True)
with col4:
    st.markdown("**Instructions**")
    st.write("- Modifie S₀ / N dans la barre latérale pour recalculer la surface.")
    st.write("- Cette surface montre comment le prix binomial dépend du spot et du nombre d'étapes.")

# -----------------------
# Delta hedge simulation (Monte Carlo)
# -----------------------
if run_hedge:
    st.subheader("🛡️ Simulation de couverture Δ (modèle binomial)")

    st.markdown(
        "Procédure :\n"
        "- Pour chaque simulation MC, on simule un chemin GBM du sous-jacent avec M = `rebalancing_N` pas.\n"
        "- À chaque pas t_i on calcule Δ_binomial(S_t, K, r, σ, T - t_i, N_remaining)\n"
        "- On réplique l'option en détenant Δ actions et un montant en cash (financé au taux r), "
        "on rééquilibre à chaque pas.\n"
        "- On calcule le P&L final du portefeuille répliquant moins le payoff de l'option.\n"
        "On affiche la moyenne et la distribution."
    )

    # MC simulation parameters
    M = int(rebalancing_N)
    sims = int(n_sims)
    dt = T / M
    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)
    rng = np.random.default_rng(12345)

    pnl_list = []
    for sim in range(sims):
        # simulate GBM path
        Zs = rng.normal(size=M)
        S_path = np.empty(M + 1)
        S_path[0] = S0
        for t in range(1, M + 1):
            S_path[t] = S_path[t - 1] * np.exp(drift + vol * Zs[t - 1])

        # initial option price and delta (using full N=M)
        opt_price = binomial_option_price(S_path[0], K, r, sigma, T, M, option_type)
        delta = binomial_delta(S_path[0], K, r, sigma, T, M, option_type)
        # replicate: hold delta shares, borrow/lend B so that portfolio = option price
        B = opt_price - delta * S_path[0]

        # iterate rebalancing
        for t in range(1, M + 1):
            # evolve cash at risk-free
            B = B * np.exp(r * dt)
            # update value of holdings before rebalancing
            # compute new delta for remaining steps
            remaining_T = T - (t * dt)
            remaining_N = M - t
            if remaining_N <= 0:
                new_delta = 0.0
            else:
                new_delta = binomial_delta(S_path[t], K, r, sigma, remaining_T, remaining_N, option_type)
            # rebalance: adjust shares, financing the difference
            # cost to change delta: (new_delta - delta) * S_t
            B = B - (new_delta - delta) * S_path[t]
            delta = new_delta

        # at maturity: portfolio value = delta_final * S_T + B - option_payoff (we are replicator)
        if option_type == "Call":
            payoff = max(S_path[-1] - K, 0.0)
        else:
            payoff = max(K - S_path[-1], 0.0)
        portfolio_value = delta * S_path[-1] + B
        pnl = portfolio_value - payoff
        pnl_list.append(pnl)

    pnl_arr = np.array(pnl_list)
    mean_pnl = pnl_arr.mean()
    std_pnl = pnl_arr.std()

    colA, colB = st.columns(2)
    with colA:
        st.metric("P&L moyen du hedge (MC)", f"{mean_pnl:.6f}")
        st.metric("Écart-type du P&L", f"{std_pnl:.6f}")
    with colB:
        st.write("Histogramme du P&L (100 bins)")
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=pnl_arr, nbinsx=100))
        hist_fig.update_layout(title="Distribution du P&L de la couverture Δ",
                               xaxis_title="P&L", yaxis_title="Fréquence", template="plotly_white", height=380)
        st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown(
        "⭐ Interprétation : si la réplication est parfaite (N grand et rééquilibrage fréquent), "
        "la moyenne du P&L devrait être proche de 0 (on réplique l'option). Les imperfections viennent "
        "du nombre limité d'étapes binomiales et du caractère discret du rééquilibrage."
    )

st.markdown("---")
st.markdown("✅ *Astuce* : si tu veux que j'ajoute la visualisation de Δ(t) le long d'un chemin exemple, "
            "ou un export CSV des résultats MC, dis-le moi et je l'intègre.")
