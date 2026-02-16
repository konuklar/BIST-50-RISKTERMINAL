import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="BIST 50 Risk Budgeting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .warning-text {
        color: #DC2626;
        font-weight: 600;
    }
    .success-text {
        color: #059669;
        font-weight: 600;
    }
    .data-source-badge {
        background-color: #1E3A8A;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# Analyzer
# ============================================================
class BIST50RiskAnalyzer:
    """Risk Budgeting Analysis for BIST 50 Stocks using Yahoo Finance"""

    # Yahoo Finance BIST tickers (verified working format)
    tickers = [
        'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'EKGYO.IS',
        'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'HALKB.IS', 'ISCTR.IS',
        'KCHOL.IS', 'KOZAL.IS', 'KRDMD.IS', 'PETKM.IS', 'PGSUS.IS',
        'SAHOL.IS', 'SASA.IS', 'TCELL.IS', 'THYAO.IS', 'TOASO.IS'
    ]

    # Company names
    asset_names = {
        'AKBNK.IS': 'Akbank',
        'ARCLK.IS': 'Arcelik',
        'ASELS.IS': 'Aselsan',
        'BIMAS.IS': 'BIM',
        'EKGYO.IS': 'Emlak Konut',
        'EREGL.IS': 'Eregli Demir Celik',
        'FROTO.IS': 'Ford Otosan',
        'GARAN.IS': 'Garanti BBVA',
        'HALKB.IS': 'Halkbank',
        'ISCTR.IS': 'Is Bankasi',
        'KCHOL.IS': 'Koc Holding',
        'KOZAL.IS': 'Koza Altin',
        'KRDMD.IS': 'Kardemir',
        'PETKM.IS': 'Petkim',
        'PGSUS.IS': 'Pegasus',
        'SAHOL.IS': 'Sabanci Holding',
        'SASA.IS': 'SASA Polyester',
        'TCELL.IS': 'Turkcell',
        'THYAO.IS': 'Turkish Airlines',
        'TOASO.IS': 'Tofas'
    }

    # Sectors
    sectors = {
        'AKBNK.IS': 'Banking',
        'ARCLK.IS': 'Industrial',
        'ASELS.IS': 'Defense',
        'BIMAS.IS': 'Retail',
        'EKGYO.IS': 'Real Estate',
        'EREGL.IS': 'Iron & Steel',
        'FROTO.IS': 'Automotive',
        'GARAN.IS': 'Banking',
        'HALKB.IS': 'Banking',
        'ISCTR.IS': 'Banking',
        'KCHOL.IS': 'Holding',
        'KOZAL.IS': 'Mining',
        'KRDMD.IS': 'Iron & Steel',
        'PETKM.IS': 'Petrochemical',
        'PGSUS.IS': 'Aviation',
        'SAHOL.IS': 'Holding',
        'SASA.IS': 'Chemicals',
        'TCELL.IS': 'Telecom',
        'THYAO.IS': 'Aviation',
        'TOASO.IS': 'Automotive'
    }

    def __init__(self):
        self.data_loaded = False

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_yahoo_data(tickers, start_date, end_date, min_obs=60):
        """
        Fetch data from Yahoo Finance robustly.
        Notes:
        - With auto_adjust=True, 'Close' is typically the adjusted close proxy.
        - Handles multi-index structures safely.
        """
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        data = yf.download(
            tickers=tickers,
            start=start_str,
            end=end_str,
            progress=False,
            group_by='ticker',
            auto_adjust=True,
            timeout=30,
            threads=True
        )

        if data is None or getattr(data, "empty", True):
            return None, None, "No data received from Yahoo Finance."

        # Extract adjusted prices robustly
        prices = None

        # MultiIndex case (common when multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            # Typical with group_by='ticker': (Ticker, Field)
            lvl0 = data.columns.get_level_values(0)
            lvl1 = data.columns.get_level_values(1)

            if "Close" in lvl1:
                prices = data.xs("Close", axis=1, level=1)
            elif "Adj Close" in lvl1:
                prices = data.xs("Adj Close", axis=1, level=1)
            else:
                # Fallback: take last level if possible
                # Try to find any price-like field
                for candidate in ["Close", "Adj Close", "Price", "Last"]:
                    if candidate in lvl1:
                        prices = data.xs(candidate, axis=1, level=1)
                        break

                if prices is None:
                    # Last resort: try to coerce
                    prices = data.copy()

        else:
            # Single ticker or non-multiindex structure
            if isinstance(data, pd.DataFrame):
                if "Close" in data.columns:
                    prices = data["Close"]
                elif "Adj Close" in data.columns:
                    prices = data["Adj Close"]
                else:
                    # If only one column-like structure exists, fallback
                    prices = data.iloc[:, 0]
            else:
                # Series
                prices = data

        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        # Ensure columns are tickers (if possible)
        # Sometimes xs returns columns already as tickers; if not, attempt to normalize
        # Drop completely empty columns
        prices = prices.dropna(axis=1, how="all")

        if prices.empty:
            return None, None, "Price extraction failed (empty price matrix)."

        # Cleaning
        prices = prices.replace([np.inf, -np.inf], np.nan)
        prices = prices.fillna(method="ffill", limit=3)
        prices = prices.dropna(how="any")

        # Remove illiquid/short-history tickers
        valid_cols = [c for c in prices.columns if prices[c].dropna().shape[0] >= min_obs]
        prices = prices[valid_cols]

        if prices.shape[1] < 2:
            return None, None, "Not enough valid tickers after cleaning (need at least 2)."

        returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")

        if returns.empty or len(returns) < min_obs:
            return None, None, "Returns series too short after cleaning."

        return prices, returns, None

    def calculate_risk_metrics(self, returns, weights=None):
        """
        Calculate comprehensive risk metrics for a portfolio.

        FIX: supports custom weights (risk parity) instead of always resetting to equal-weight.
        """
        cols = list(returns.columns)
        n_assets = len(cols)

        if weights is None:
            w = np.ones(n_assets) / n_assets
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape[0] != n_assets:
                raise ValueError("weights length must match number of assets")
            if np.any(~np.isfinite(w)):
                raise ValueError("weights contain non-finite values")
            # normalize
            w = np.clip(w, 0, None)
            w = w / (w.sum() if w.sum() != 0 else 1.0)

        # Annualized covariance (252)
        cov_matrix = returns.cov() * 252
        cov_np = cov_matrix.values

        portfolio_variance = float(w @ cov_np @ w)
        portfolio_volatility = float(np.sqrt(max(portfolio_variance, 0)))

        indiv_vol = np.sqrt(np.diag(cov_np))

        # Marginal risk contributions (MRC)
        # mrc_i = (Sigma w)_i / sigma_p
        if portfolio_volatility > 0:
            marginal_risk = (cov_np @ w) / portfolio_volatility
        else:
            marginal_risk = np.zeros(n_assets)

        component_risk = w * marginal_risk
        pct_contributions = (component_risk / portfolio_volatility * 100) if portfolio_volatility > 0 else np.zeros(n_assets)

        risk_metrics = pd.DataFrame({
            "Symbol": cols,
            "Company": [self.asset_names.get(t, t) for t in cols],
            "Sector": [self.sectors.get(t, "Other") for t in cols],
            "Weight": w,
            "Individual_Volatility": indiv_vol,
            "Marginal_Risk_Contribution": marginal_risk,
            "Component_Risk": component_risk,
            "Risk_Contribution_%": pct_contributions
        })

        risk_metrics = risk_metrics.sort_values("Risk_Contribution_%", ascending=False).reset_index(drop=True)
        risk_metrics["Risk_Rank"] = np.arange(1, len(risk_metrics) + 1)

        # Betas vs portfolio
        portfolio_returns = returns.values @ w
        port_var_daily = float(np.var(portfolio_returns, ddof=1))
        betas = []
        for i, col in enumerate(cols):
            asset_ret = returns[col].values
            cov_daily = float(np.cov(asset_ret, portfolio_returns, ddof=1)[0, 1])
            beta = (cov_daily / port_var_daily) if port_var_daily > 0 else np.nan
            betas.append(beta)

        risk_metrics["Beta"] = betas

        weighted_avg_vol = float(np.sum(w * indiv_vol))
        diversification_ratio = (weighted_avg_vol / portfolio_volatility) if portfolio_volatility > 0 else np.nan

        portfolio_metrics = {
            "volatility": portfolio_volatility,
            "diversification_ratio": diversification_ratio,
            "n_assets": n_assets,
            "avg_volatility": float(np.mean(indiv_vol)),
            "max_risk_contrib": float(risk_metrics.iloc[0]["Risk_Contribution_%"]),
            "max_risk_asset": str(risk_metrics.iloc[0]["Company"])
        }

        return risk_metrics, portfolio_metrics, cov_matrix

    def calculate_risk_parity(self, cov_matrix, returns_columns):
        """Calculate risk parity weights (SLSQP)."""
        cols = list(returns_columns)
        n = len(cols)
        cov_np = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix)

        def risk_parity_objective(weights):
            weights = np.asarray(weights, dtype=float)
            weights = np.clip(weights, 0, None)
            weights = weights / (weights.sum() if weights.sum() != 0 else 1.0)

            portfolio_var = float(weights @ cov_np @ weights)
            portfolio_vol = float(np.sqrt(max(portfolio_var, 0)))

            if portfolio_vol <= 0:
                return 1e9

            marginal_contrib = (cov_np @ weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib

            target_rc = portfolio_vol / n
            rc_deviation = risk_contrib - target_rc
            return float(np.sum(rc_deviation ** 2))

        init_weights = np.ones(n) / n
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(n)]

        result = minimize(
            risk_parity_objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 2000}
        )

        if result.success and np.all(np.isfinite(result.x)):
            w = np.clip(result.x, 0, None)
            return w / (w.sum() if w.sum() != 0 else 1.0)
        else:
            return init_weights


# ============================================================
# Main App
# ============================================================
def main():
    st.markdown('<p class="main-header">📊 BIST 50 Risk Budgeting Dashboard</p>', unsafe_allow_html=True)

    st.markdown("""
    This dashboard analyzes **Marginal Risk Contributions (MRC)** and provides **Risk Budgeting**
    recommendations for an equally weighted portfolio of 20 major BIST 50 stocks.
    """)

    st.markdown("""
    <div class="data-source-badge">
        📡 Data Source: Yahoo Finance (Real-time)
    </div>
    """, unsafe_allow_html=True)

    analyzer = BIST50RiskAnalyzer()

    # Sidebar
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Borsa_Istanbul_logo.svg/200px-Borsa_Istanbul_logo.svg.png",
            width=150
        )

        st.markdown("## ⚙️ Parameters")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime(2020, 1, 1),
                max_value=datetime.now() - timedelta(days=30)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.now(),
                max_value=datetime.now()
            )

        if start_date >= end_date:
            st.error("Start date must be before end date")
            st.stop()

        portfolio_type = st.radio(
            "Portfolio Type",
            ["Equal Weight", "Risk Parity (Optimized)"],
            help="Equal weight portfolio or risk parity optimization"
        )

        st.markdown("### 📈 Display Options")
        show_individual_vol = st.checkbox("Individual Volatility", value=True)
        show_mrc = st.checkbox("Marginal Risk Contribution (MRC)", value=True)
        show_beta = st.checkbox("Beta Coefficients", value=True)

        st.markdown("---")
        st.markdown(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.markdown("**Status:** 🟢 Active")

    # Main content
    try:
        with st.spinner("📥 Fetching real-time data from Yahoo Finance..."):
            prices, returns, err = analyzer.fetch_yahoo_data(analyzer.tickers, start_date, end_date)

        if err:
            st.error(err)
            st.stop()

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trading Days", len(returns))
        with col2:
            st.metric("Stocks Loaded", len(returns.columns))
        with col3:
            period_days = (end_date - start_date).days
            st.metric("Period (Days)", period_days)

        # Portfolio calculations
        if portfolio_type == "Equal Weight":
            risk_metrics, portfolio_metrics, cov_matrix = analyzer.calculate_risk_metrics(returns, weights=None)
            active_weights = np.ones(len(returns.columns)) / len(returns.columns)
        else:
            # Step 1: get covariance
            tmp_rm, tmp_pm, cov_matrix = analyzer.calculate_risk_metrics(returns, weights=None)
            # Step 2: risk parity weights
            rp_weights = analyzer.calculate_risk_parity(cov_matrix, returns.columns)
            # Step 3: compute metrics using RP weights (FIXED)
            risk_metrics, portfolio_metrics, cov_matrix = analyzer.calculate_risk_metrics(returns, weights=rp_weights)
            active_weights = rp_weights

        # Key metrics
        st.markdown('<p class="sub-header">📌 Key Portfolio Metrics</p>', unsafe_allow_html=True)
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)

        with mcol1:
            st.metric("Portfolio Volatility (Annual)", f"{portfolio_metrics['volatility']:.2%}")
        with mcol2:
            st.metric("Average Individual Vol", f"{portfolio_metrics['avg_volatility']:.2%}")
        with mcol3:
            st.metric("Diversification Ratio", f"{portfolio_metrics['diversification_ratio']:.2f}",
                      help=">1.5 indicates good diversification")
        with mcol4:
            st.metric("Top Risk Contributor", portfolio_metrics["max_risk_asset"], f"{portfolio_metrics['max_risk_contrib']:.1f}%")

        # Risk Contribution Analysis
        st.markdown('<p class="sub-header">🎯 Risk Contribution Analysis</p>', unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns([2, 1])

        with chart_col1:
            fig = go.Figure()
            sorted_df = risk_metrics.sort_values("Risk_Contribution_%", ascending=True)

            fig.add_trace(go.Bar(
                y=sorted_df["Company"],
                x=sorted_df["Risk_Contribution_%"],
                orientation="h",
                marker=dict(
                    color=sorted_df["Risk_Contribution_%"],
                    colorscale="RdYlGn_r",
                    showscale=True,
                    colorbar=dict(title="Risk %")
                ),
                text=sorted_df["Risk_Contribution_%"].round(1).astype(str) + "%",
                textposition="outside",
                name="Risk Contribution"
            ))

            equal_contrib = 100 / len(sorted_df)
            fig.add_vline(
                x=equal_contrib,
                line_dash="dash",
                line_color="red",
                opacity=0.7,
                annotation_text=f"Equal Risk Target ({equal_contrib:.1f}%)"
            )

            fig.update_layout(
                title="Risk Contribution by Asset (Ranked)",
                xaxis_title="Risk Contribution (%)",
                yaxis_title="",
                height=600,
                showlegend=False,
                hovermode="y"
            )
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            top_3 = sorted_df.tail(3)["Risk_Contribution_%"].sum()
            top_5 = sorted_df.tail(5)["Risk_Contribution_%"].sum()
            others = 100 - top_5

            fig = go.Figure(data=[go.Pie(
                labels=["Top 3 Contributors", "Next 2 Contributors", "Remaining"],
                values=[top_3, top_5 - top_3, others],
                hole=0.4,
                marker_colors=["#DC2626", "#F59E0B", "#10B981"]
            )])

            fig.update_layout(
                title="Risk Concentration Analysis",
                annotations=[dict(
                    text=f"Top 3: {top_3:.1f}%",
                    x=0.5, y=0.5,
                    font_size=14,
                    showarrow=False
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detailed Metrics Table (numeric-safe for ProgressColumn)
        st.markdown('<p class="sub-header">📋 Detailed Risk Metrics</p>', unsafe_allow_html=True)

        base_cols = ["Risk_Rank", "Company", "Sector", "Weight", "Risk_Contribution_%"]
        if show_individual_vol:
            base_cols.append("Individual_Volatility")
        if show_mrc:
            base_cols.append("Marginal_Risk_Contribution")
        if show_beta:
            base_cols.append("Beta")

        display_df = risk_metrics[base_cols].copy()

        # Rename for UI
        display_df = display_df.rename(columns={
            "Risk_Rank": "Rank",
            "Risk_Contribution_%": "Risk %",
            "Individual_Volatility": "Indiv Vol",
            "Marginal_Risk_Contribution": "MRC"
        })

        # Streamlit column configs (keep numeric)
        column_config = {
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Weight": st.column_config.NumberColumn("Weight", format="%.2%"),
            "Risk %": st.column_config.ProgressColumn(
                "Risk Contribution",
                help="Percentage of total portfolio risk",
                min_value=0.0,
                max_value=100.0,
                format="%.1f%%"
            )
        }

        if show_individual_vol:
            column_config["Indiv Vol"] = st.column_config.NumberColumn("Indiv Vol", format="%.2%")
        if show_mrc:
            column_config["MRC"] = st.column_config.NumberColumn("MRC", format="%.5f")
        if show_beta:
            column_config["Beta"] = st.column_config.NumberColumn("Beta", format="%.2f")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )

        # Risk Parity Recommendations (only for equal weight mode)
        if portfolio_type == "Equal Weight":
            st.markdown('<p class="sub-header">⚖️ Risk Parity Recommendations</p>', unsafe_allow_html=True)

            rp_weights = analyzer.calculate_risk_parity(cov_matrix, returns.columns)

            recommendations = []
            n = len(returns.columns)
            for i, symbol in enumerate(returns.columns):
                current_w = 1 / n
                rp_w = rp_weights[i]
                adjustment = rp_w - current_w

                company = analyzer.asset_names.get(symbol, symbol)
                sector = analyzer.sectors.get(symbol, "Other")

                action = "REDUCE" if adjustment < -0.002 else "INCREASE" if adjustment > 0.002 else "MAINTAIN"

                recommendations.append({
                    "Company": company,
                    "Sector": sector,
                    "Current Weight": current_w,
                    "Risk Parity Weight": rp_w,
                    "Adjustment": adjustment,
                    "Action": action
                })

            rec_df = pd.DataFrame(recommendations)

            def color_action(val):
                if val == "REDUCE":
                    return "background-color: #FEE2E2; color: #DC2626"
                elif val == "INCREASE":
                    return "background-color: #DCFCE7; color: #059669"
                else:
                    return "background-color: #F3F4F6; color: #6B7280"

            styled_rec = rec_df.style.applymap(color_action, subset=["Action"]).format({
                "Current Weight": "{:.2%}",
                "Risk Parity Weight": "{:.2%}",
                "Adjustment": "{:+.2%}"
            })

            st.dataframe(styled_rec, use_container_width=True, hide_index=True)

            csv = rec_df.copy()
            csv["Current Weight"] = (csv["Current Weight"] * 100).round(4)
            csv["Risk Parity Weight"] = (csv["Risk Parity Weight"] * 100).round(4)
            csv["Adjustment"] = (csv["Adjustment"] * 100).round(4)

            st.download_button(
                label="📥 Download Recommendations (CSV)",
                data=rec_df.to_csv(index=False),
                file_name=f"risk_parity_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        # Sector analysis (FIX: correct weight percentage)
        st.markdown('<p class="sub-header">🏭 Sector Risk Analysis</p>', unsafe_allow_html=True)

        sector_analysis = risk_metrics.groupby("Sector", as_index=True).agg({
            "Risk_Contribution_%": "sum",
            "Weight": "sum"
        }).rename(columns={
            "Risk_Contribution_%": "Total Risk %",
            "Weight": "Total Weight"
        })

        sector_analysis["Total Weight %"] = sector_analysis["Total Weight"] * 100
        sector_analysis = sector_analysis.drop(columns=["Total Weight"])
        sector_analysis = sector_analysis.sort_values("Total Risk %", ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Risk Contribution",
            x=sector_analysis.index,
            y=sector_analysis["Total Risk %"],
            marker_color="#EF4444"
        ))
        fig.add_trace(go.Bar(
            name="Portfolio Weight",
            x=sector_analysis.index,
            y=sector_analysis["Total Weight %"],
            marker_color="#3B82F6"
        ))
        fig.update_layout(
            title="Sector Risk vs Weight Allocation",
            xaxis_title="Sector",
            yaxis_title="Percentage (%)",
            barmode="group",
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export report (in-memory Excel)
        if st.button("📊 Generate Full Report"):
            buffer = BytesIO()

            # Prefer xlsxwriter if available; fallback to openpyxl
            engine = None
            try:
                import xlsxwriter  # noqa: F401
                engine = "xlsxwriter"
            except Exception:
                engine = "openpyxl"

            with pd.ExcelWriter(buffer, engine=engine) as writer:
                risk_metrics.to_excel(writer, sheet_name="Risk Metrics", index=False)
                sector_analysis.reset_index().to_excel(writer, sheet_name="Sector Analysis", index=False)

                if portfolio_type == "Equal Weight":
                    # Create RP rec sheet only when applicable
                    # (rec_df exists only in that branch)
                    try:
                        rec_df.to_excel(writer, sheet_name="Recommendations", index=False)
                    except Exception:
                        pass

            buffer.seek(0)

            st.download_button(
                label="📥 Download Full Report (Excel)",
                data=buffer,
                file_name=f"bist50_risk_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")


if __name__ == "__main__":
    main()
