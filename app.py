import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import binomial_distribution as bdis
import geometric_distribution as gdis
import poisson_distribution as pdis

# Page configuration
st.set_page_config(
    page_title="Irrigation Probability Advisory System", page_icon="💧", layout="wide"
)


# ---------------------------------------------------------
# Data Loading & Ingestion
# ---------------------------------------------------------
@st.cache_data
def load_telemetry(path: str = "./data/dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


try:
    df = load_telemetry("./data/dataset.csv")
except Exception:
    df = load_telemetry("dataset.csv")

# ---------------------------------------------------------
# Sidebar Parameter Configurator
# ---------------------------------------------------------
st.sidebar.header("Operational Thresholds")

temp_threshold = st.sidebar.slider(
    "Max Temperature Trigger (°C)",
    min_value=float(df["max_temperature"].min()),
    max_value=float(df["max_temperature"].max()),
    value=38.0,
    step=0.2,
)

precip_threshold = st.sidebar.slider(
    "Precipitation Cutoff (mm)",
    min_value=float(df["precipitation"].min()),
    max_value=float(df["precipitation"].max()),
    value=0.10,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.header("System Capacity & Planning")

planning_window = st.sidebar.slider(
    "Forward Planning Horizon n (Days)", min_value=3, max_value=21, value=7, step=1
)

capacity_limit = st.sidebar.slider(
    "Peak Pumping Capacity Threshold (Days)",
    min_value=1,
    max_value=planning_window,
    value=max(1, planning_window // 2),
    step=1,
)

# ---------------------------------------------------------
# Empirical Parameter Estimation
# ---------------------------------------------------------
df["irrigation_required"] = (df["max_temperature"] > temp_threshold) & (
    df["precipitation"] < precip_threshold
)

total_days = len(df)
total_events = int(df["irrigation_required"].sum())
p = total_events / total_days if total_days > 0 else 0.0

st.title("Irrigation Probability Advisory System")
st.caption(
    "Statistical Process Modeling: Binomial, Poisson, and Geometric Distributions"
)

# High-level telemetry metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Historical Analyzed Days", f"{total_days} days")
m2.metric("Triggered Irrigation Events", f"{total_events} events")
m3.metric("Baseline Daily Risk (p)", f"{p:.4f}")
m4.metric("Arrival Rate (λ over n days)", f"{(planning_window * p):.2f}")

# Guard against extreme probability inputs outside distribution domains
if p <= 0:
    st.error(
        "No irrigation events match current thresholds (p = 0). Lower the temperature threshold or raise the precipitation limit."
    )
    st.stop()
elif p > 1:
    st.error("Invalid probability domain (p > 1). Adjust environmental filters.")
    st.stop()

# ---------------------------------------------------------
# Tabbed Engineering Analysis
# ---------------------------------------------------------
tab_telemetry, tab_binomial, tab_poisson, tab_geometric, tab_advisory = st.tabs(
    [
        "Telemetry & Correlation",
        "Binomial Capacity",
        "Poisson Arrivals",
        "Geometric Wait Time",
        "Advisory Matrix",
    ]
)

# ---------------------------------------------------------
# TAB 1: Telemetry & Correlation
# ---------------------------------------------------------
with tab_telemetry:
    st.subheader("Sensor Time Series & Parameter Correlation")

    col_t1, col_t2 = st.columns([3, 2])

    with col_t1:
        fig_ts, ax_temp = plt.subplots(figsize=(10, 4.5))
        ax_rain = ax_temp.twinx()

        days_axis = df["date_of_year"]

        # Dual-axis environmental telemetry plot
        ax_temp.plot(
            days_axis,
            df["max_temperature"],
            color="#d9534f",
            linewidth=1.8,
            label="Max Temp (°C)",
        )
        ax_temp.axhline(
            temp_threshold,
            color="#d9534f",
            linestyle="--",
            alpha=0.7,
            label=f"Temp Trigger ({temp_threshold}°C)",
        )
        ax_temp.set_ylabel("Max Temperature (°C)", color="#d9534f")
        ax_temp.tick_params(axis="y", labelcolor="#d9534f")

        ax_rain.bar(
            days_axis,
            df["precipitation"],
            color="#337ab7",
            alpha=0.35,
            width=0.6,
            label="Precipitation (mm)",
        )
        ax_rain.set_ylabel("Precipitation (mm)", color="#337ab7")
        ax_rain.tick_params(axis="y", labelcolor="#337ab7")

        # Flag trigger occurrences
        triggered_days = df[df["irrigation_required"]]
        ax_temp.scatter(
            triggered_days["date_of_year"],
            triggered_days["max_temperature"],
            color="#2e6da4",
            s=45,
            zorder=5,
            label="Irrigation Trigger",
        )

        ax_temp.set_xlabel("Day of Year")
        ax_temp.set_title("Environmental Time Series & Active Trigger Thresholds")
        ax_temp.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig_ts)
        plt.close(fig_ts)

    with col_t2:
        # Correlation Heatmap
        corr_vars = [
            "max_temperature",
            "precipitation",
            "root_zone_soil_wetness",
            "irrigation_required",
        ]
        corr_matrix = df[corr_vars].astype(float).corr()

        fig_corr, ax_corr = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
            ax=ax_corr,
            square=True,
            linewidths=0.5,
        )
        ax_corr.set_title("Variable Correlation Matrix")
        st.pyplot(fig_corr)
        plt.close(fig_corr)

# ---------------------------------------------------------
# TAB 2: Binomial Capacity Planning
# ---------------------------------------------------------
with tab_binomial:
    st.subheader(f"Binomial Capacity Planning (Horizon n = {planning_window} Days)")

    n_range = list(range(planning_window + 1))
    bin_probs = []

    for k in n_range:
        prob_k, b_mean, b_var, b_sd = bdis.binomial(n=planning_window, p=p, x=k)
        bin_probs.append(prob_k)

    df_binom = pd.DataFrame(
        {
            "Required_Days_k": n_range,
            "Probability": bin_probs,
            "Exceeds_Capacity": [k > capacity_limit for k in n_range],
        }
    )

    col_b1, col_b2 = st.columns([3, 1])

    with col_b1:
        fig_binom, ax_b = plt.subplots(figsize=(9, 4.5))
        palette = [
            "#d9534f" if exc else "#337ab7" for exc in df_binom["Exceeds_Capacity"]
        ]

        ax_b.bar(
            df_binom["Required_Days_k"],
            df_binom["Probability"],
            color=palette,
            edgecolor="black",
            alpha=0.85,
        )
        ax_b.axvline(
            capacity_limit + 0.5,
            color="#d9534f",
            linestyle="--",
            linewidth=1.5,
            label=f"Capacity Limit ({capacity_limit} days)",
        )
        ax_b.set_xlabel("Number of Irrigation Days (k)")
        ax_b.set_ylabel("Probability P(X = k)")
        ax_b.set_title("Binomial PMF: Schedule Allocation & Overload Boundary")
        ax_b.set_xticks(n_range)
        ax_b.grid(axis="y", linestyle=":", alpha=0.6)
        ax_b.legend()
        st.pyplot(fig_binom)
        plt.close(fig_binom)

    with col_b2:
        risk_exceed = sum(df_binom.loc[df_binom["Exceeds_Capacity"], "Probability"])
        st.metric("Expected Cycles E[X]", f"{b_mean:.2f} days")
        st.metric("Cycle Standard Dev", f"{b_sd:.2f} days")
        st.metric("Capacity Risk P(X > k_cap)", f"{risk_exceed:.4f}")

        if risk_exceed > 0.25:
            st.error(
                f"High Overload Risk: {risk_exceed:.1%} chance of exceeding pump capacity."
            )
        else:
            st.success(f"Nominal Risk: Pump sizing covers planned horizon.")

# ---------------------------------------------------------
# TAB 3: Poisson Event Arrivals
# ---------------------------------------------------------
with tab_poisson:
    st.subheader("Poisson Arrival Rate Approximation")

    lam = planning_window * p
    pois_range = list(range(planning_window + 1))
    pois_probs = [pdis.poisson_probability(lam=lam, y=y) for y in pois_range]

    df_compare = pd.DataFrame(
        {"Events_k": pois_range, "Binomial": bin_probs, "Poisson": pois_probs}
    )

    fig_comp, ax_comp = plt.subplots(figsize=(10, 4.5))
    x_indices = np.arange(len(pois_range))
    bar_width = 0.35

    ax_comp.bar(
        x_indices - bar_width / 2,
        df_compare["Binomial"],
        width=bar_width,
        label="Binomial (True Finite)",
        color="#337ab7",
        alpha=0.85,
    )
    ax_comp.bar(
        x_indices + bar_width / 2,
        df_compare["Poisson"],
        width=bar_width,
        label="Poisson (Arrival Rate λ)",
        color="#f0ad4e",
        alpha=0.85,
    )

    ax_comp.set_xlabel("Number of Events (k)")
    ax_comp.set_ylabel("Probability Mass")
    ax_comp.set_title(f"Binomial vs. Poisson Convergence (λ = {lam:.2f})")
    ax_comp.set_xticks(x_indices)
    ax_comp.set_xticklabels(pois_range)
    ax_comp.grid(axis="y", linestyle=":", alpha=0.6)
    ax_comp.legend()
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    st.caption(
        "Poisson models random event arrivals over time. Deviation occurs when sample size n is small and base event probability p is non-negligible."
    )

# ---------------------------------------------------------
# TAB 4: Geometric Dry-Down & Waiting Time
# ---------------------------------------------------------
with tab_geometric:
    st.subheader("Geometric Distribution: Waiting Time to Next Trigger")

    max_eval_days = 12
    geom_data = gdis.calculate_geometric_irrigation_wait(p=p, max_days=max_eval_days)
    df_geom = geom_data["summary_table"]

    col_g1, col_g2 = st.columns([3, 1])

    with col_g1:
        fig_geom, ax_g1 = plt.subplots(figsize=(9, 4.5))
        ax_g2 = ax_g1.twinx()

        # PMF bars
        ax_g1.bar(
            df_geom["Consecutive_Days_k"],
            df_geom["Probability_Pk"],
            color="#5cb85c",
            alpha=0.75,
            edgecolor="black",
            label="PMF P(Z = k)",
        )
        ax_g1.set_xlabel("Dry Days Until First Irrigation Event (k)")
        ax_g1.set_ylabel("PMF Probability", color="#4cae4c")
        ax_g1.tick_params(axis="y", labelcolor="#4cae4c")

        # CDF line
        ax_g2.plot(
            df_geom["Consecutive_Days_k"],
            df_geom["Cumulative_Probability"],
            color="#204d74",
            marker="o",
            linewidth=2.0,
            label="CDF P(Z ≤ k)",
        )
        ax_g2.set_ylabel("Cumulative Probability", color="#204d74")
        ax_g2.tick_params(axis="y", labelcolor="#204d74")
        ax_g2.set_ylim(0, 1.05)

        ax_g1.set_title("Geometric Waiting Time: PMF & Hazard Depletion CDF")
        ax_g1.grid(axis="x", linestyle=":", alpha=0.6)
        st.pyplot(fig_geom)
        plt.close(fig_geom)

    with col_g2:
        st.metric("Expected Wait E[Z]", f"{geom_data['expected_wait_days']:.2f} days")
        st.metric("Variance Var(Z)", f"{geom_data['variance']:.2f}")
        st.metric("Standard Dev", f"{np.sqrt(geom_data['variance']):.2f} days")

# ---------------------------------------------------------
# TAB 5: Engineering Advisory Matrix
# ---------------------------------------------------------
with tab_advisory:
    st.subheader("Automated Operational Directives")

    expected_wait = geom_data["expected_wait_days"]
    high_capacity_risk = (
        sum(df_binom.loc[df_binom["Exceeds_Capacity"], "Probability"]) > 0.25
    )

    st.markdown("### Operational Recommendations")
    if high_capacity_risk:
        st.warning(
            f"**Action Required: Water Allocation Exceedance.**\n"
            f"The probability of exceeding pumping capacity ({capacity_limit} days) over the next {planning_window} days is {sum(df_binom.loc[df_binom['Exceeds_Capacity'], 'Probability']):.2%}. "
            "Pre-fill secondary storage buffers or throttle line delivery."
        )
    else:
        st.success(
            f"**Capacity Nominal.**\n"
            f"Pumping capacity limit of {capacity_limit} days carries a safe buffer. Exceedance probability is below target threshold."
        )

    st.info(
        f"**Preventative Maintenance Window:**\n"
        f"Expected dry-spell lead time is {expected_wait:.1f} days. Complete valve inspection and line checks within {max(1, int(expected_wait))} days of last rain."
    )

    # Export report capability
    csv_export = df_binom.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Binomial Risk Matrix (CSV)",
        data=csv_export,
        file_name="irrigation_binomial_risk.csv",
        mime="text/csv",
    )
