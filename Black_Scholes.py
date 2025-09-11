import streamlit as st
import numpy as np
from scipy.stats import norm

st.title("⚖️ Black-Scholes Calculator")

# Inputs
S = st.number_input("Spot Price", value=100.0)
K = st.number_input("Strike Price", value=100.0)
T = st.number_input("Time to Maturity (in years)", value=1.0)
r = st.number_input("Risk-Free Rate (decimal)", value=0.05)
sigma = st.number_input("Volatility (decimal)", value=0.2)

# Formula
d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)

call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
put = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

st.metric("📈 Call Option Price", f"{call:.2f}")
st.metric("📉 Put Option Price", f"{put:.2f}")
