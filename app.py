import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Executive FP&A Revenue & Risk Forecasting Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Executive FP&A Revenue Forecasting & What-If Analysis Engine")
st.markdown("An AI-Powered Financial Planning & Analysis (FP&A) tool for Revenue Forecasting, Monte Carlo Risk Simulation, and P&L Planning.")

st.divider()

# ---------------------------------------------------------
# 2. Synthetic Data Generation & ML Model Training
# ---------------------------------------------------------
@st.cache_data
def generate_and_train():
    np.random.seed(42)
    n_months = 36
    
    # Historical Drivers
    mkt_spend = np.random.uniform(20000, 100000, n_months)
    price_per_unit = np.random.uniform(80, 150, n_months)
    churn_rate = np.random.uniform(2.0, 8.0, n_months)
    inflation_rate = np.random.uniform(1.0, 5.0, n_months)
    
    # Revenue Formula with realistic noise
    revenue = (
        (mkt_spend * 8.5) + 
        (price_per_unit * 1200) - 
        (churn_rate * 5000) - 
        (inflation_rate * 2000) + 
        np.random.normal(0, 15000, n_months)
    )
    
    df = pd.DataFrame({
        'Marketing_Spend': mkt_spend,
        'Price_Per_Unit': price_per_unit,
        'Churn_Rate_%': churn_rate,
        'Inflation_Rate_%': inflation_rate,
        'Revenue': revenue
    })
    
    # ML Model
    X = df[['Marketing_Spend', 'Price_Per_Unit', 'Churn_Rate_%', 'Inflation_Rate_%']]
    y = df['Revenue']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return df, model

df_history, ml_model = generate_and_train()

# ---------------------------------------------------------
# 3. Sidebar: Scenarios & Drivers
# ---------------------------------------------------------
st.sidebar.header("🎛️ FP&A Driver Inputs")

scenario = st.sidebar.selectbox(
    "Select Business Scenario",
    ["Custom / Manual Input", "Bull Case (Optimistic)", "Bear Case (Pessimistic)"]
)

# Preset logic for scenarios
if scenario == "Bull Case (Optimistic)":
    default_mkt = 85000.0
    default_price = 135.0
    default_churn = 2.5
    default_inflation = 2.0
elif scenario == "Bear Case (Pessimistic)":
    default_mkt = 30000.0
    default_price = 90.0
    default_churn = 7.5
    default_inflation = 5.0
else:
    default_mkt = 50000.0
    default_price = 110.0
    default_churn = 4.0
    default_inflation = 3.0

st.sidebar.subheader("Adjust Drivers")
mkt_input = st.sidebar.slider("Marketing Spend ($)", 10000.0, 150000.0, default_mkt, step=5000.0)
price_input = st.sidebar.slider("Price Per Unit ($)", 50.0, 200.0, default_price, step=5.0)
churn_input = st.sidebar.slider("Churn Rate (%)", 1.0, 10.0, default_churn, step=0.5)
inf_input = st.sidebar.slider("Inflation Rate (%)", 1.0, 8.0, default_inflation, step=0.5)

st.sidebar.subheader("Cost Structure (P&L Assumptions)")
cogs_pct = st.sidebar.slider("COGS (% of Revenue)", 20, 60, 35) / 100.0
fixed_opex = st.sidebar.number_input("Fixed OPEX ($ / Year)", value=150000, step=10000)

# ---------------------------------------------------------
# 4. Core Calculations (ML + FP&A)
# ---------------------------------------------------------
input_features = np.array([[mkt_input, price_input, churn_input, inf_input]])
predicted_monthly_rev = ml_model.predict(input_features)[0]
projected_annual_rev = predicted_monthly_rev * 12

# P&L Calculations
cogs = projected_annual_rev * cogs_pct
gross_profit = projected_annual_rev - cogs
total_opex = fixed_opex + (mkt_input * 12)
ebitda = gross_profit - total_opex
ebitda_margin = (ebitda / projected_annual_rev) * 100 if projected_annual_rev > 0 else 0

# Break-Even Calculations
contribution_margin_ratio = 1 - cogs_pct
breakeven_revenue = total_opex / contribution_margin_ratio if contribution_margin_ratio > 0 else 0
breakeven_units = breakeven_revenue / price_input if price_input > 0 else 0

# ---------------------------------------------------------
# 5. Top KPI Summary Cards
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Forecasted Annual Revenue", f"${projected_annual_rev:,.0f}")
kpi2.metric("Gross Profit", f"${gross_profit:,.0f}", f"{(1 - cogs_pct)*100:.1f}% Margin")
kpi3.metric("Projected EBITDA", f"${ebitda:,.0f}", f"{ebitda_margin:.1f}% EBITDA Margin")
kpi4.metric("Break-Even Revenue", f"${breakeven_revenue:,.0f}")

st.divider()

# ---------------------------------------------------------
# 6. Main Content Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📉 P&L & Break-Even Analysis", "🎲 Monte Carlo Risk Simulation", "📝 Executive Summary & Export"])

with tab1:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📑 Projected Income Statement (P&L)")
        pnl_data = {
            "Financial Line Item": [
                "Gross Revenue", 
                "(-) Cost of Goods Sold (COGS)", 
                "Gross Profit", 
                "(-) Fixed OPEX", 
                "(-) Marketing Expense", 
                "EBITDA (Operating Profit)"
            ],
            "Amount ($)": [
                f"${projected_annual_rev:,.2f}",
                f"-${cogs:,.2f}",
                f"${gross_profit:,.2f}",
                f"-${fixed_opex:,.2f}",
                f"-${mkt_input*12:,.2f}",
                f"${ebitda:,.2f}"
            ]
        }
        pnl_df = pd.DataFrame(pnl_data)
        st.table(pnl_df)
        
    with col_right:
        st.subheader("⚖️ Break-Even Analysis Plot")
        fig_be, ax_be = plt.subplots(figsize=(6, 4))
        
        revenue_range = np.linspace(0, projected_annual_rev * 1.5, 100)
        total_costs = fixed_opex + (mkt_input * 12) + (revenue_range * cogs_pct)
        
        ax_be.plot(revenue_range, revenue_range, label="Total Revenue", color="green", lw=2)
        ax_be.plot(revenue_range, total_costs, label="Total Cost", color="red", lw=2)
        ax_be.axvline(x=breakeven_revenue, color="gray", linestyle="--", label=f"Break-Even (${breakeven_revenue:,.0f})")
        
        ax_be.set_xlabel("Revenue ($)")
        ax_be.set_ylabel("Cost / Revenue ($)")
        ax_be.set_title("Revenue vs. Total Costs")
        ax_be.legend()
        ax_be.grid(True, alpha=0.3)
        st.pyplot(fig_be)

with tab2:
    st.subheader("🎲 Monte Carlo Risk Simulation (1,000 Scenarios)")
    st.markdown("Simulating market volatility (±15% variance in demand and marketing efficiency) to find confidence intervals.")
    
    simulations = 1000
    simulated_revenues = []
    
    for _ in range(simulations):
        mkt_sim = np.random.normal(mkt_input, mkt_input * 0.1)
        price_sim = np.random.normal(price_input, price_input * 0.05)
        churn_sim = np.random.normal(churn_input, churn_input * 0.1)
        inf_sim = np.random.normal(inf_input, inf_input * 0.05)
        
        rev_sim = ml_model.predict([[mkt_sim, price_sim, churn_sim, inf_sim]])[0] * 12
        simulated_revenues.append(rev_sim)
        
    simulated_revenues = np.array(simulated_revenues)
    
    p5 = np.percentile(simulated_revenues, 5)
    p50 = np.percentile(simulated_revenues, 50)
    p95 = np.percentile(simulated_revenues, 95)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Worst Case (5th Percentile)", f"${p5:,.0f}")
    c2.metric("Base Case (Median)", f"${p50:,.0f}")
    c3.metric("Best Case (95th Percentile)", f"${p95:,.0f}")
    
    fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
    ax_mc.hist(simulated_revenues, bins=40, color="skyblue", edgecolor="black", alpha=0.7)
    ax_mc.axvline(p5, color="red", linestyle="--", label="P5 (Bear Case Risk)")
    ax_mc.axvline(p50, color="black", linestyle="-", label="P50 (Median)")
    ax_mc.axvline(p95, color="green", linestyle="--", label="P95 (Bull Case Upside)")
    ax_mc.set_title("Annual Revenue Distribution Outcome")
    ax_mc.set_xlabel("Projected Revenue ($)")
    ax_mc.set_ylabel("Frequency")
    ax_mc.legend()
    st.pyplot(fig_mc)

with tab3:
    st.subheader("🧠 Automated Managerial Commentary")
    
    comments = []
    if ebitda < 0:
        comments.append("🚨 **CRITICAL ALERT:** Projected EBITDA is **Negative**. Immediate cost reduction or price restructuring required.")
    elif ebitda_margin < 15:
        comments.append("⚠️ **MARGIN WARNING:** EBITDA Margin is below 15%. Consider optimizing OPEX or reducing marketing spend.")
    else:
        comments.append("✅ **HEALTHY MARGINS:** EBITDA Margin is robust (> 15%). The business unit is generating healthy cash flows.")
        
    if projected_annual_rev < breakeven_revenue:
        comments.append("🔴 **DEFICIT RISK:** Forecasted revenue is below the Break-Even point. The company will incur operating losses.")
    else:
        comments.append(f"🟢 **SURPLUS:** Forecasted revenue exceeds Break-Even point by **${projected_annual_rev - breakeven_revenue:,.0f}**.")
        
    if churn_input > 5.0:
        comments.append("📉 **HIGH CHURN:** Churn rate is high (> 5%). Retention programs should be prioritized over customer acquisition.")
        
    for comment in comments:
        st.write(comment)
        
    st.divider()
    
    st.subheader("📥 Export Financial Report")
    export_df = pd.DataFrame({
        "Metric": ["Scenario", "Annual Revenue", "COGS", "Gross Profit", "Total OPEX", "EBITDA", "Break-Even Revenue"],
        "Value": [scenario, projected_annual_rev, cogs, gross_profit, total_opex, ebitda, breakeven_revenue]
    })
    
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download FP&A Summary as CSV",
        data=csv,
        file_name="fpa_executive_summary.csv",
        mime="text/csv"
    )