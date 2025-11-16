import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# ===============================
# 📊 Configuration de la page
# ===============================
st.set_page_config(page_title="Binomial vs Black-Scholes", page_icon="📈", layout="wide")

st.title("📊 Convergence du modèle binomial vers Black-Scholes")
st.markdown("""
Cette application montre *le passage du modèle discret (Binomial) vers le modèle continu (Black–Scholes)*.
""")

# ===============================
# ⚙️ Entrées utilisateur
# ===============================
col1, col2, col3 = st.columns(3)
with col1:
    S0 = st.number_input("Prix initial de l’action (S₀)", value=100.0, min_value=0.0)
    K = st.number_input("Prix d’exercice (K)", value=100.0, min_value=0.0)
with col2:
    r = st.number_input("Taux sans risque (r)", value=0.05, min_value=0.0, step=0.01)
    sigma = st.number_input("Volatilité (σ)", value=0.2, min_value=0.0, step=0.01)
with col3:
    T = st.number_input("Maturité (T, en années)", value=1.0, min_value=0.01)
    option_type = st.selectbox("Type d’option", ["Call", "Put"])

# ===============================
# 📘 Fonctions de calcul
# ===============================
def black_scholes(S0, K, r, sigma, T, option_type):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

def binomial_option_price(S0, K, r, sigma, T, N, option_type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    ST = S0 * d ** np.arange(N, -1, -1) * u ** np.arange(0, N + 1)
    if option_type == "Call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    for i in range(N, 0, -1):
        payoff = disc * (q * payoff[1:] + (1 - q) * payoff[:-1])
    return payoff[0]

# ===============================
# 📈 Calcul et visualisation
# ===============================
st.divider()
st.subheader("🔢 Simulation")

max_steps = st.slider("Nombre maximal d’étapes binomiales", 10, 1000, 100)
steps_range = np.arange(1, max_steps + 1, max_steps // 50 or 1)

binomial_prices = [binomial_option_price(S0, K, r, sigma, T, N, option_type) for N in steps_range]
bs_price = black_scholes(S0, K, r, sigma, T, option_type)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=steps_range, y=binomial_prices,
    mode='lines+markers', name='Prix Binomial',
    line=dict(color='royalblue', width=2)
))
fig.add_trace(go.Scatter(
    x=steps_range, y=[bs_price]*len(steps_range),
    mode='lines', name='Prix Black-Scholes',
    line=dict(color='orange', dash='dash', width=2)
))
fig.update_layout(
    title="Convergence du prix binomial vers Black–Scholes",
    xaxis_title="Nombre d’étapes (N)",
    yaxis_title="Prix de l’option",
    template="plotly_white",
    legend=dict(x=0.02, y=0.98)
)
st.plotly_chart(fig, use_container_width=True)

# ===============================
# 🧾 Résultats numériques
# ===============================
st.subheader("📋 Résultats")
colA, colB = st.columns(2)
with colA:
    st.metric(label="Prix Black-Scholes", value=f"{bs_price:.4f}")
with colB:
    st.metric(label=f"Prix Binomial (N = {max_steps})", value=f"{binomial_prices[-1]:.4f}")

st.markdown("""
✅ *Observation :* Quand le nombre d’étapes \(N\) augmente, le prix binomial *converge* vers le prix continu Black–Scholes.
""")
