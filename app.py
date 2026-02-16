import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta
from io import BytesIO
import time
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 📊 BIST 50 Risk Budgeting Dashboard
# - Robust Yahoo Finance fetching (retry + per-ticker fallback)
# - Correct Risk Parity application (custom weights supported)
# - Numeric-safe Streamlit ProgressColumn
# - Correct sector weight percentages
# - Streamlit Cloud-safe Excel export (BytesIO, xlsxwriter/openpyxl fallback)
# ============================================================

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="BIST 50 Risk Budgeting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        font-weight: 600;
        margin-top: 0.75rem;
        margin-bottom: 0.35rem;
    }
    .data-source-badge {
        background-color: #1E3A8A;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 0.75rem;
    }
    .small-note {
        color: #6B7280;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# Core Analyzer
# ============================================================
class BIST50RiskAnalyzer:
    """Risk Budgeting Analysis for BIST 50 Stocks using Yahoo Finance"""

    # Yahoo Finance BIST tickers (working format)
    tickers = [
        "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS",
        "EREGL.IS", "FROTO.IS", "GARAN.IS", "HALKB.IS", "ISCTR.IS",
        "KCHOL.IS", "KOZAL.IS", "KRDMD.IS", "PETKM.IS", "PGSUS.IS",
        "SAHOL.IS", "SASA.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS"
    ]

    asset_names = {
        "AKBNK.IS": "Akbank",
        "ARCLK.IS": "Arcelik",
        "ASELS.IS": "Aselsan",
        "BIMAS.IS": "BIM",
        "EKGYO.IS": "Emlak Konut",
        "EREGL.IS": "Eregli Demir Celik",
        "FROTO.IS": "Ford Otosan",
        "GARAN.IS": "Garanti BBVA",
        "HALKB.IS": "Halkbank",
        "ISCTR.IS": "Is Bankasi",
        "KCHOL.IS": "Koc Holding",
        "KOZAL.IS": "Koza Altin",
        "KRDMD.IS": "Kardemir",
        "PETKM.IS": "Petkim",
        "PGSUS.IS": "Pegasus",
        "SAHOL.IS": "Sabanci Holding",
        "SASA.IS": "SASA Polyester",
        "TCELL.IS": "Turkcell",
        "THYAO.IS": "Turkish Airlines",
        "TOASO.IS": "Tofas"
    }

    sectors = {
        "AKBNK.IS": "Banking",
        "ARCLK.IS": "Industrial",
        "ASELS.IS": "Defense",
        "BIMAS.IS": "Retail",
        "EKGYO.IS": "Real Estate",
        "EREGL.IS": "Iron & Steel",
        "FROTO.IS": "Automotive",
        "GARAN.IS": "Banking",
        "HALKB.IS": "Banking",
        "ISCTR.IS": "Banking",
        "KCHOL.IS": "Holding",
        "KOZAL.IS": "Mining",
        "KRDMD.IS": "Iron & Steel",
        "PETKM.IS": "Petrochemical",
        "PGSUS.IS": "Aviation",
        "SAHOL.IS": "Holding",
        "SASA.IS": "Chemicals",
        "TCELL.IS": "Telecom",
        "THYAO.IS": "Aviation",
        "TOASO.IS": "Automotive"
    }

    def __init__(self):
        pass

    # ------------------------------------------------------------
    # Robust Yahoo Finance data fetcher
    # - tries batch download (with retries)
    # - falls back to per-ticker history()
    # - cleans prices and returns
    # ------------------------------------------------------------
    @staticmethod
    @st.cache_data(ttl=1800, show_spinner=False)
    def fetch_yahoo_data(tickers, start_date, end_date, max_retries=3, pause=1.5, min_obs=60):
        """
        Fetch daily adjusted prices from Yahoo Finance.

        Why this robust version?
        - Yahoo endpoints can return EMPTY responses intermittently (or rate-limit cloud IPs).
        - Batch download is efficient, but can fail more often on shared hosting.
        - Per-ticker fallback is slower, but often succeeds.

        Returns:
            prices (DataFrame), returns (DataFrame), err (str or None)

        Side-channel:
            returns.attrs["failed_tickers"] contains tickers that were dropped in fallback.
        """
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        def _clean_prices(prices: pd.DataFrame):
            prices = prices.replace([np.inf, -np.inf], np.nan)
            prices = prices.dropna(axis=1, how="all")

            # Forward fill small holiday gaps (limit to 3 sessions)
            prices = prices.ffill(limit=3)

            # Drop rows still containing NaN
            prices = prices.dropna(how="any")

            # Keep only tickers with enough history
            good = [c for c in prices.columns if prices[c].dropna().shape[0] >= min_obs]
            prices = prices[good]

            if prices.shape[1] < 2:
                return None, None, "Not enough valid tickers after cleaning (need at least 2)."

            rets = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")

            if rets.empty or len(rets) < min_obs:
                return None, None, "Returns series too short after cleaning."

            return prices, rets, None

        # -----------------------
        # 1) Batch download w/ retries
        # -----------------------
        last_err = None
        data = pd.DataFrame()

        for k in range(max_retries):
            try:
                data = yf.download(
                    tickers=tickers,
                    start=start_str,
                    end=end_str,
                    interval="1d",
                    auto_adjust=True,
                    group_by="column",   # typical output: MultiIndex (Field, Ticker)
                    progress=False,
                    threads=False,       # threads=True can increase empty responses in some environments
                    timeout=30
                )
                if data is not None and not data.empty:
                    break
            except Exception as e:
                last_err = e
            time.sleep(pause * (2 ** k))

        if data is not None and not data.empty:
            # Parse batch structure
            if isinstance(data.columns, pd.MultiIndex):
                lvl0 = data.columns.get_level_values(0)
                if "Close" in lvl0:
                    prices = data["Close"].copy()
                elif "Adj Close" in lvl0:
                    prices = data["Adj Close"].copy()
                else:
                    return None, None, "Batch download succeeded but no Close/Adj Close fields found."
            else:
                # Single ticker or unusual structure
                if "Close" in data.columns:
                    prices = data[["Close"]].rename(columns={"Close": tickers[0]})
                elif "Adj Close" in data.columns:
                    prices = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
                else:
                    return None, None, "Batch download succeeded but no Close/Adj Close columns found."

            return _clean_prices(prices)

        # -----------------------
        # 2) Per-ticker fallback
        # -----------------------
        frames = {}
        failed = []

        for t in tickers:
            try:
                hist = yf.Ticker(t).history(
                    start=start_str,
                    end=end_str,
                    interval="1d",
                    auto_adjust=True
                )
                if hist is None or hist.empty or "Close" not in hist.columns:
                    failed.append(t)
                    continue
                frames[t] = hist["Close"].rename(t)
            except Exception:
                failed.append(t)

            # gentle pacing helps with rate limits
            time.sleep(0.25)

        if not frames:
            msg = "No data received from Yahoo Finance (batch + per-ticker both failed). "
            if last_err:
                msg += f"Last error: {last_err}. "
            msg += "This is typically Yahoo blocking/rate-limiting your IP/environment (common on shared cloud IPs)."
            return None, None, msg

        prices = pd.concat(frames.values(), axis=1).sort_index()
        prices, returns, err = _clean_prices(prices)
        if err:
            return None, None, err

        returns.attrs["failed_tickers"] = failed
        return prices, returns, None

    # ------------------------------------------------------------
    # Risk metrics: supports custom weights (Equal Weight / RP)
    # ------------------------------------------------------------
    def calculate_risk_metrics(self, returns: pd.DataFrame, weights=None):
        """
        Computes:
        - annualized covariance (252)
        - portfolio vol
        - individual vols
        - MRC/CRC and risk contribution %
        - beta vs portfolio
        - diversification ratio
        """
        cols = list(returns.columns)
        n_assets = len(cols)

        # weights
        if weights is None:
            w = np.ones(n_assets) / n_assets
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape[0] != n_assets:
                raise ValueError("weights length must match number of assets")
            if np.any(~np.isfinite(w)):
                raise ValueError("weights contain non-finite values")
            w = np.clip(w, 0, None)
            w = w / (w.sum() if w.sum() != 0 else 1.0)

        cov_df = returns.cov() * 252
        cov = cov_df.values

        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(max(port_var, 0)))

        indiv_vol = np.sqrt(np.diag(cov))

        if port_vol > 0:
            mrc = (cov @ w) / port_vol
            crc = w * mrc
            rc_pct = (crc / port_vol) * 100
        else:
            mrc = np.zeros(n_assets)
            crc = np.zeros(n_assets)
            rc_pct = np.zeros(n_assets)

        risk_metrics = pd.DataFrame({
            "Symbol": cols,
            "Company": [self.asset_names.get(t, t) for t in cols],
            "Sector": [self.sectors.get(t, "Other") for t in cols],
            "Weight": w,
            "Individual_Volatility": indiv_vol,
            "Marginal_Risk_Contribution": mrc,
            "Component_Risk": crc,
            "Risk_Contribution_%": rc_pct
        }).sort_values("Risk_Contribution_%", ascending=False).reset_index(drop=True)

        risk_metrics["Risk_Rank"] = np.arange(1, len(risk_metrics) + 1)

        # Beta vs portfolio
        port_ret = returns.values @ w
        port_var_d = float(np.var(port_ret, ddof=1))

        betas = []
        for col in cols:
            a = returns[col].values
            cov_d = float(np.cov(a, port_ret, ddof=1)[0, 1])
            beta = (cov_d / port_var_d) if port_var_d > 0 else np.nan
            betas.append(beta)

        risk_metrics["Beta"] = betas

        weighted_avg_vol = float(np.sum(w * indiv_vol))
        div_ratio = (weighted_avg_vol / port_vol) if port_vol > 0 else np.nan

        portfolio_metrics = {
            "volatility": port_vol,
            "diversification_ratio": div_ratio,
            "n_assets": n_assets,
            "avg_volatility": float(np.mean(indiv_vol)),
            "max_risk_contrib": float(risk_metrics.iloc[0]["Risk_Contribution_%"]),
            "max_risk_asset": str(risk_metrics.iloc[0]["Company"]),
        }

        return risk_metrics, portfolio_metrics, cov_df

    # ------------------------------------------------------------
    # Risk parity (SLSQP)
    # ------------------------------------------------------------
    def calculate_risk_parity(self, cov_matrix: pd.DataFrame, returns_columns):
        """
        Find weights that equalize component risk contributions.

        Objective: minimize sum((RC_i - target)^2)
        """
        cols = list(returns_columns)
        n = len(cols)
        cov = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix)

        def objective(x):
            x = np.asarray(x, dtype=float)
            x = np.clip(x, 0, None)
            x = x / (x.sum() if x.sum() != 0 else 1.0)

            var = float(x @ cov @ x)
            vol = float(np.sqrt(max(var, 0)))

            if vol <= 0:
                return 1e9

            mrc = (cov @ x) / vol
            rc = x * mrc
            target = vol / n
            return float(np.sum((rc - target) ** 2))

        x0 = np.ones(n) / n
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(n)]

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 2000}
        )

        if res.success and np.all(np.isfinite(res.x)):
            w = np.clip(res.x, 0, None)
            return w / (w.sum() if w.sum() != 0 else 1.0)
        return x0


# ============================================================
# App
# ============================================================
def main():
    st.markdown('<p class="main-header">📊 BIST 50 Risk Budgeting Dashboard</p>', unsafe_allow_html=True)

    st.markdown("""
    This dashboard analyzes **Marginal Risk Contributions (MRC)** and provides **Risk Budgeting**
    insights for a portfolio of major BIST stocks.

    <div class="small-note">
    Note: Yahoo Finance may occasionally return empty data due to rate limits (especially on shared/cloud IPs).
    This app uses retries + per-ticker fallback to improve reliability.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="data-source-badge">
        📡 Data Source: Yahoo Finance (Real-time)
    </div>
    """, unsafe_allow_html=True)

    analyzer = BIST50RiskAnalyzer()

    # ------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Borsa_Istanbul_logo.svg/200px-Borsa_Istanbul_logo.svg.png",
            width=150
        )

        st.markdown("## ⚙️ Parameters")

        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input(
                "Start Date",
                datetime(2020, 1, 1),
                max_value=datetime.now() - timedelta(days=30)
            )
        with c2:
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
        st.caption(f"yfinance version: {yf.__version__}")

        with st.expander("🧪 Yahoo Health Check", expanded=False):
            try:
                test = yf.download("XU100.IS", period="5d", interval="1d", progress=False, threads=False)
                rows = 0 if (test is None or test.empty) else len(test)
                st.write("Rows:", rows)
                if rows > 0:
                    st.dataframe(test.tail(3), use_container_width=True)
                else:
                    st.warning("Health check returned 0 rows. Yahoo may be blocking/rate-limiting this environment.")
            except Exception as e:
                st.error(str(e))

        st.markdown("---")
        st.markdown(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.markdown("**Status:** 🟢 Active")

    # ------------------------------------------------------------
    # Fetch data
    # ------------------------------------------------------------
    try:
        with st.spinner("📥 Fetching real-time data from Yahoo Finance..."):
            prices, returns, err = analyzer.fetch_yahoo_data(analyzer.tickers, start_date, end_date)

        if err:
            st.error(err)
            st.stop()

        failed = returns.attrs.get("failed_tickers", [])
        if failed:
            st.warning(f"Yahoo did not return data for {len(failed)} tickers (dropped): {failed}")

        # ------------------------------------------------------------
        # Summary Metrics
        # ------------------------------------------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trading Days", len(returns))
        with col2:
            st.metric("Stocks Loaded", len(returns.columns))
        with col3:
            st.metric("Period (Days)", (end_date - start_date).days)

        # ------------------------------------------------------------
        # Portfolio metrics
        # ------------------------------------------------------------
        if portfolio_type == "Equal Weight":
            risk_metrics, portfolio_metrics, cov_matrix = analyzer.calculate_risk_metrics(returns, weights=None)
        else:
            # Step 1: covariance from equal weights
            _, _, cov_matrix = analyzer.calculate_risk_metrics(returns, weights=None)

            # Step 2: risk parity weights
            rp_w = analyzer.calculate_risk_parity(cov_matrix, returns.columns)

            # Step 3: compute with rp weights (CRITICAL FIX)
            risk_metrics, portfolio_metrics, cov_matrix = analyzer.calculate_risk_metrics(returns, weights=rp_w)

        # ------------------------------------------------------------
        # Key Portfolio Metrics
        # ------------------------------------------------------------
        st.markdown('<p class="sub-header">📌 Key Portfolio Metrics</p>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Portfolio Volatility (Annual)", f"{portfolio_metrics['volatility']:.2%}")
        with m2:
            st.metric("Average Individual Vol", f"{portfolio_metrics['avg_volatility']:.2%}")
        with m3:
            st.metric("Diversification Ratio", f"{portfolio_metrics['diversification_ratio']:.2f}",
                      help=">1.5 indicates good diversification")
        with m4:
            st.metric("Top Risk Contributor", portfolio_metrics["max_risk_asset"],
                      f"{portfolio_metrics['max_risk_contrib']:.1f}%")

        # ------------------------------------------------------------
        # Risk Contribution Analysis
        # ------------------------------------------------------------
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

            equal_target = 100 / len(sorted_df)
            fig.add_vline(
                x=equal_target,
                line_dash="dash",
                line_color="red",
                opacity=0.7,
                annotation_text=f"Equal Risk Target ({equal_target:.1f}%)"
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
            others = max(0.0, 100 - top_5)

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

        # ------------------------------------------------------------
        # Detailed Metrics Table (numeric safe)
        # ------------------------------------------------------------
        st.markdown('<p class="sub-header">📋 Detailed Risk Metrics</p>', unsafe_allow_html=True)

        cols = ["Risk_Rank", "Company", "Sector", "Weight", "Risk_Contribution_%"]
        if show_individual_vol:
            cols.append("Individual_Volatility")
        if show_mrc:
            cols.append("Marginal_Risk_Contribution")
        if show_beta:
            cols.append("Beta")

        display_df = risk_metrics[cols].copy().rename(columns={
            "Risk_Rank": "Rank",
            "Risk_Contribution_%": "Risk %",
            "Individual_Volatility": "Indiv Vol",
            "Marginal_Risk_Contribution": "MRC"
        })

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
            column_config["MRC"] = st.column_config.NumberColumn("MRC", format="%.6f")
        if show_beta:
            column_config["Beta"] = st.column_config.NumberColumn("Beta", format="%.2f")

        st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=column_config)

        # ------------------------------------------------------------
        # Risk Parity Recommendations (shown only in Equal Weight mode)
        # ------------------------------------------------------------
        if portfolio_type == "Equal Weight":
            st.markdown('<p class="sub-header">⚖️ Risk Parity Recommendations</p>', unsafe_allow_html=True)

            rp_w = analyzer.calculate_risk_parity(cov_matrix, returns.columns)

            recs = []
            n = len(returns.columns)
            for i, sym in enumerate(returns.columns):
                cur_w = 1 / n
                new_w = rp_w[i]
                adj = new_w - cur_w

                action = "REDUCE" if adj < -0.002 else "INCREASE" if adj > 0.002 else "MAINTAIN"

                recs.append({
                    "Company": analyzer.asset_names.get(sym, sym),
                    "Sector": analyzer.sectors.get(sym, "Other"),
                    "Current Weight": cur_w,
                    "Risk Parity Weight": new_w,
                    "Adjustment": adj,
                    "Action": action
                })

            rec_df = pd.DataFrame(recs)

            def color_action(val):
                if val == "REDUCE":
                    return "background-color: #FEE2E2; color: #DC2626"
                if val == "INCREASE":
                    return "background-color: #DCFCE7; color: #059669"
                return "background-color: #F3F4F6; color: #6B7280"

            styled = rec_df.style.applymap(color_action, subset=["Action"]).format({
                "Current Weight": "{:.2%}",
                "Risk Parity Weight": "{:.2%}",
                "Adjustment": "{:+.2%}"
            })

            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.download_button(
                label="📥 Download Recommendations (CSV)",
                data=rec_df.to_csv(index=False),
                file_name=f"risk_parity_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        # ------------------------------------------------------------
        # Sector Analysis (FIX: correct weights as %)
        # ------------------------------------------------------------
        st.markdown('<p class="sub-header">🏭 Sector Risk Analysis</p>', unsafe_allow_html=True)

        sector_analysis = (
            risk_metrics.groupby("Sector", as_index=True)
            .agg({"Risk_Contribution_%": "sum", "Weight": "sum"})
            .rename(columns={"Risk_Contribution_%": "Total Risk %", "Weight": "Total Weight"})
        )

        sector_analysis["Total Weight %"] = sector_analysis["Total Weight"] * 100
        sector_analysis = sector_analysis.drop(columns=["Total Weight"]).sort_values("Total Risk %", ascending=False)

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
            height=420,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------------------
        # Full Report Export (Excel in-memory)
        # ------------------------------------------------------------
        st.markdown('<p class="sub-header">📦 Export</p>', unsafe_allow_html=True)

        if st.button("📊 Generate Full Report"):
            buffer = BytesIO()

            # Prefer xlsxwriter if available; fallback to openpyxl
            engine = "openpyxl"
            try:
                import xlsxwriter  # noqa: F401
                engine = "xlsxwriter"
            except Exception:
                engine = "openpyxl"

            with pd.ExcelWriter(buffer, engine=engine) as writer:
                risk_metrics.to_excel(writer, sheet_name="Risk Metrics", index=False)
                sector_analysis.reset_index().to_excel(writer, sheet_name="Sector Analysis", index=False)

                # If recommendations exist in scope, export them too
                if portfolio_type == "Equal Weight":
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
