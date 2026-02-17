
# =============================================================
# 📊 BIST Risk Budgeting Dashboard (Streamlit Community Cloud)
# Single-file app.py — robust Yahoo downloader + constraints
# =============================================================
#
# Features
# - Universe:
#     * Default 20 (curated, KOZAL->TRALT fix)
#     * AUTO_BIST50 (scrape OYAK XU050 table)
#     * Custom list (text area)
# - Portfolio:
#     * Equal Weight
#     * Constrained Risk Parity (max weight, min weight, optional sector caps)
# - Risk:
#     * Risk contributions (MRC/CRC/%RC)
#     * VaR / ES(CVaR): historical, parametric, modified (Cornish–Fisher)
#     * Rolling risk contributions (%RC over time)
# - Exports:
#     * CSV downloads
#     * Excel report (in-memory)
#
# Notes for BIST on Yahoo:
# - Yahoo can intermittently return missing / sparse series.
# - This app uses best-effort cleaning (no full date-intersection requirement)
#   and covariance-safe pruning to avoid "not enough tickers" failures.

from __future__ import annotations

import io
import time
import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Tuple, Dict, List, Union, Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

st.set_page_config(
    page_title="BIST Risk Budgeting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.main-header { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.4rem; }
.sub-header { font-size: 1.35rem; font-weight: 700; margin: 0.75rem 0 0.25rem 0; }
.badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
         background: #0f2a5f; color: white; font-size: 0.85rem; margin: 0.2rem 0 0.6rem 0; }
.small-note { color: #6b7280; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

def normalize_tickers(tickers: Iterable[str], suffix: str = ".IS") -> List[str]:
    out = []
    for t in tickers:
        if t is None:
            continue
        s = str(t).strip().upper()
        if not s:
            continue
        if s.startswith("^"):
            out.append(s)
            continue
        if "." in s:
            out.append(s)
            continue
        out.append(s + suffix)

    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq

def parse_ticker_input(ticker_input: Union[str, List[str], Tuple[str, ...], None]) -> Optional[List[str]]:
    if ticker_input is None:
        return None
    if isinstance(ticker_input, (list, tuple)):
        return normalize_tickers(ticker_input)
    s = str(ticker_input).strip()
    if not s:
        return None
    if s.upper() == "AUTO_BIST50":
        return ["AUTO_BIST50"]
    parts = [p.strip() for p in s.replace(",", "\n").splitlines()]
    parts = [p for p in parts if p]
    return normalize_tickers(parts)

def apply_ticker_aliases(tickers: List[str]) -> Tuple[List[str], Dict[str, str]]:
    alias = {"KOZAL.IS": "TRALT.IS", "KOZAL": "TRALT.IS"}
    mapping_used = {}
    mapped = []
    for t in tickers:
        t0 = t
        t1 = alias.get(t0, t0)
        if t1 != t0:
            mapping_used[t0] = t1
        mapped.append(t1)
    mapped = normalize_tickers(mapped)
    return mapped, mapping_used

def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.clip(w, 0, None)
    w = w / (w.sum() if w.sum() != 0 else 1.0)
    pr = returns.values @ w
    return pd.Series(pr, index=returns.index, name="Portfolio")

@dataclass(frozen=True)
class AssetMeta:
    name: str
    sector: str

class BISTRiskAnalyzer:
    default_20: List[str] = [
        "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS",
        "EREGL.IS", "FROTO.IS", "GARAN.IS", "HALKB.IS", "ISCTR.IS",
        "KCHOL.IS", "TRALT.IS", "KRDMD.IS", "PETKM.IS", "PGSUS.IS",
        "SAHOL.IS", "SASA.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS",
    ]

    base_meta: Dict[str, AssetMeta] = {
        "AKBNK.IS": AssetMeta("Akbank", "Banking"),
        "ARCLK.IS": AssetMeta("Arcelik", "Industrial"),
        "ASELS.IS": AssetMeta("Aselsan", "Defense"),
        "BIMAS.IS": AssetMeta("BIM", "Retail"),
        "EKGYO.IS": AssetMeta("Emlak Konut", "Real Estate"),
        "EREGL.IS": AssetMeta("Eregli Demir Celik", "Iron & Steel"),
        "FROTO.IS": AssetMeta("Ford Otosan", "Automotive"),
        "GARAN.IS": AssetMeta("Garanti BBVA", "Banking"),
        "HALKB.IS": AssetMeta("Halkbank", "Banking"),
        "ISCTR.IS": AssetMeta("Is Bankasi", "Banking"),
        "KCHOL.IS": AssetMeta("Koc Holding", "Holding"),
        "KOZAL.IS": AssetMeta("Koza Altin (Legacy)", "Mining"),
        "TRALT.IS": AssetMeta("Turk Altin Isletmeleri", "Mining"),
        "KRDMD.IS": AssetMeta("Kardemir", "Iron & Steel"),
        "PETKM.IS": AssetMeta("Petkim", "Petrochemical"),
        "PGSUS.IS": AssetMeta("Pegasus", "Aviation"),
        "SAHOL.IS": AssetMeta("Sabanci Holding", "Holding"),
        "SASA.IS": AssetMeta("SASA Polyester", "Chemicals"),
        "TCELL.IS": AssetMeta("Turkcell", "Telecom"),
        "THYAO.IS": AssetMeta("Turkish Airlines", "Aviation"),
        "TOASO.IS": AssetMeta("Tofas", "Automotive"),
    }

    def __init__(self):
        self.meta: Dict[str, AssetMeta] = dict(self.base_meta)

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_bist50_tickers_online(timeout: int = 20) -> List[str]:
        url = "https://www.oyakyatirim.com.tr/piyasa-verileri/XU050"
        try:
            tables = pd.read_html(url, flavor="lxml")
            df = tables[0]
            sym_col = None
            for c in df.columns:
                if str(c).strip().lower() in ("sembol", "symbol", "kod", "code"):
                    sym_col = c
                    break
            if sym_col is None:
                sym_col = df.columns[0]
            symbols = df[sym_col].astype(str).str.strip().tolist()
            return normalize_tickers(symbols, suffix=".IS")
        except Exception:
            import requests
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            tables = pd.read_html(r.text, flavor="lxml")
            df = tables[0]
            sym_col = "Sembol" if "Sembol" in df.columns else df.columns[0]
            symbols = df[sym_col].astype(str).str.strip().tolist()
            return normalize_tickers(symbols, suffix=".IS")

    @staticmethod
    def yahoo_health_check() -> Tuple[bool, str]:
        try:
            test = yf.download("XU100.IS", period="5d", interval="1d", auto_adjust=True, progress=False, threads=False)
            if test is None or test.empty:
                test2 = yf.download("THYAO.IS", period="5d", interval="1d", auto_adjust=True, progress=False, threads=False)
                if test2 is None or test2.empty:
                    return False, "Health check returned 0 rows. Yahoo may be blocking/rate-limiting this environment."
                return True, f"Health check OK (fallback THYAO.IS rows={len(test2)})."
            return True, f"Health check OK (rows={len(test)})."
        except Exception as e:
            return False, f"Health check error: {e}"

    @staticmethod
    def _best_effort_clean_panel(prices: pd.DataFrame, min_obs: int, max_missing_frac: float = 0.35):
        if prices is None or prices.empty:
            return None, None, "Empty price panel", pd.DataFrame()

        prices = prices.replace([np.inf, -np.inf], np.nan)
        prices = prices.dropna(axis=1, how="all").sort_index()
        prices = prices.ffill(limit=5)

        obs = prices.notna().sum().sort_values(ascending=False)
        diag = pd.DataFrame({
            "ObsCount": obs,
            "Start": [prices[c].first_valid_index() for c in obs.index],
            "End": [prices[c].last_valid_index() for c in obs.index],
            "MissingFrac": [float(prices[c].isna().mean()) for c in obs.index],
        })

        keep = diag.index[(diag["ObsCount"] >= min_obs) & (diag["MissingFrac"] <= max_missing_frac)].tolist()
        prices = prices[keep]

        if prices.shape[1] < 2:
            return None, None, "Not enough valid tickers after initial column filtering (need at least 2).", diag

        rets = prices.pct_change().replace([np.inf, -np.inf], np.nan)
        rets = rets.dropna(how="all")

        ret_obs = rets.notna().sum()
        keep2 = ret_obs.index[ret_obs >= (min_obs - 1)].tolist()
        rets = rets[keep2]
        prices = prices[keep2]

        if rets.shape[1] < 2:
            return None, None, "Not enough valid tickers after return filtering (need at least 2).", diag

        return prices, rets, "", diag

    @staticmethod
    def _prune_for_covariance(returns: pd.DataFrame, min_assets: int = 2, max_iter: int = 30):
        rets = returns.copy()
        for _ in range(max_iter):
            cov = rets.cov() * 252.0
            bad = cov.columns[cov.isna().any()].tolist()
            if not bad:
                return rets, cov
            rets = rets.drop(columns=bad, errors="ignore")
            if rets.shape[1] < min_assets:
                break

        rets2 = rets.dropna(how="any")
        if rets2.shape[1] >= min_assets and len(rets2) >= 30:
            cov2 = rets2.cov() * 252.0
            cov2 = cov2.dropna(axis=0, how="any").dropna(axis=1, how="any")
            keep = cov2.columns.tolist()
            return rets2[keep], cov2
        return rets, rets.cov() * 252.0

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_yahoo_data_cached(
        tickers: Tuple[str, ...],
        start_str: str,
        end_str: str,
        min_obs: int,
        max_retries: int,
    ):
        start_date = pd.to_datetime(start_str).date()
        end_date = pd.to_datetime(end_str).date()
        prices, returns, err, dropped, diag = BISTRiskAnalyzer.fetch_yahoo_data(
            list(tickers), start_date, end_date, min_obs=min_obs, max_retries=max_retries
        )
        return prices, returns, err, dropped, diag

    @staticmethod
    def fetch_yahoo_data(
        tickers: List[str],
        start_date: date,
        end_date: date,
        max_retries: int = 3,
        pause: float = 1.5,
        min_obs: int = 90,
    ):
        start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end_date).strftime("%Y-%m-%d")

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
                    group_by="column",
                    progress=False,
                    threads=False,
                    timeout=30,
                )
                if data is not None and not data.empty:
                    break
            except Exception as e:
                last_err = e
            time.sleep(pause * (2 ** k))

        prices = None
        dropped = []

        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                lvl0 = data.columns.get_level_values(0)
                if "Close" in lvl0:
                    prices = data["Close"].copy()
                elif "Adj Close" in lvl0:
                    prices = data["Adj Close"].copy()
                else:
                    return None, None, "Batch download succeeded but no Close/Adj Close found.", tickers, pd.DataFrame()
            else:
                if "Close" in data.columns:
                    prices = data[["Close"]].rename(columns={"Close": tickers[0]})
                elif "Adj Close" in data.columns:
                    prices = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
                else:
                    return None, None, "Batch download succeeded but no Close/Adj Close columns found.", tickers, pd.DataFrame()

            dropped = sorted(list(set(tickers) - set(prices.columns)))

        else:
            frames = {}
            for t in tickers:
                try:
                    hist = yf.Ticker(t).history(start=start_str, end=end_str, interval="1d", auto_adjust=True)
                    if hist is None or hist.empty or "Close" not in hist.columns:
                        dropped.append(t)
                        continue
                    frames[t] = hist["Close"].rename(t)
                except Exception:
                    dropped.append(t)
                time.sleep(0.20)

            if not frames:
                msg = "No data received from Yahoo Finance (batch + per-ticker both failed). "
                if last_err:
                    msg += f"Last error: {last_err}. "
                msg += "This is typically Yahoo blocking/rate-limiting this environment."
                return None, None, msg, dropped, pd.DataFrame()

            prices = pd.concat(frames.values(), axis=1).sort_index()

        prices_clean, rets_clean, err, diag = BISTRiskAnalyzer._best_effort_clean_panel(prices, min_obs=min_obs)
        if err:
            return None, None, err, dropped, diag

        rets_pruned, cov = BISTRiskAnalyzer._prune_for_covariance(rets_clean, min_assets=2)
        keep = rets_pruned.columns.tolist()
        prices_clean = prices_clean[keep]

        if len(keep) < 2:
            return None, None, "Not enough valid tickers after covariance pruning (need at least 2).", dropped, diag

        return prices_clean, rets_pruned, None, dropped, diag

    def calculate_risk_metrics(self, returns: pd.DataFrame, weights: Optional[np.ndarray] = None):
        cols = list(returns.columns)
        n = len(cols)

        if weights is None:
            w = np.ones(n) / n
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape[0] != n:
                raise ValueError("weights length must match number of assets")
            w = np.clip(w, 0, None)
            w = w / (w.sum() if w.sum() != 0 else 1.0)

        cov_df = returns.cov() * 252.0
        cov = cov_df.values

        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(max(port_var, 0.0)))

        indiv_vol = np.sqrt(np.clip(np.diag(cov), 0, None))

        if port_vol > 0:
            mrc = (cov @ w) / port_vol
            crc = w * mrc
            rc_pct = (crc / port_vol) * 100.0
        else:
            mrc = np.zeros(n)
            crc = np.zeros(n)
            rc_pct = np.zeros(n)

        port_ret = returns.values @ w
        port_var_d = float(np.var(port_ret, ddof=1))
        betas = []
        for c in cols:
            a = returns[c].values
            mask = np.isfinite(a) & np.isfinite(port_ret)
            if mask.sum() < 10 or port_var_d <= 0:
                betas.append(np.nan)
            else:
                cov_d = float(np.cov(a[mask], port_ret[mask], ddof=1)[0, 1])
                betas.append(cov_d / port_var_d if port_var_d > 0 else np.nan)

        wavg_vol = float(np.sum(w * indiv_vol))
        div_ratio = (wavg_vol / port_vol) if port_vol > 0 else np.nan

        rm = pd.DataFrame({
            "Symbol": cols,
            "Company": [self.meta.get(s, AssetMeta(s, "Other")).name for s in cols],
            "Sector": [self.meta.get(s, AssetMeta(s, "Other")).sector for s in cols],
            "Weight": w,
            "Individual_Volatility": indiv_vol,
            "Marginal_Risk_Contribution": mrc,
            "Component_Risk": crc,
            "Risk_Contribution_%": rc_pct,
            "Beta": betas,
        }).sort_values("Risk_Contribution_%", ascending=False).reset_index(drop=True)

        rm["Risk_Rank"] = np.arange(1, len(rm) + 1)

        pm = {
            "volatility": float(port_vol),
            "diversification_ratio": float(div_ratio),
            "n_assets": int(n),
            "avg_volatility": float(np.nanmean(indiv_vol)),
            "max_risk_contrib": float(rm.iloc[0]["Risk_Contribution_%"]),
            "max_risk_symbol": str(rm.iloc[0]["Symbol"]),
        }
        return rm, pm, cov_df

    @staticmethod
    def calculate_risk_parity_constrained(
        cov_matrix: pd.DataFrame,
        tickers: List[str],
        sectors: Optional[List[str]] = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        sector_caps: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        n = len(tickers)
        cov = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix)

        max_weight = float(max_weight)
        min_weight = float(min_weight)
        if max_weight <= 0 or max_weight > 1:
            raise ValueError("max_weight must be in (0, 1].")
        if min_weight < 0 or min_weight >= 1:
            raise ValueError("min_weight must be in [0, 1).")
        if min_weight > max_weight:
            raise ValueError("min_weight cannot exceed max_weight.")

        def obj(x):
            x = np.asarray(x, dtype=float)
            x = np.clip(x, min_weight, max_weight)
            x = x / (x.sum() if x.sum() != 0 else 1.0)

            var = float(x @ cov @ x)
            vol = float(np.sqrt(max(var, 0.0)))
            if vol <= 0:
                return 1e9
            mrc = (cov @ x) / vol
            rc = x * mrc
            target = vol / n
            return float(np.sum((rc - target) ** 2))

        x0 = np.ones(n) / n
        x0 = np.clip(x0, min_weight, max_weight)
        x0 = x0 / x0.sum()

        bounds = [(min_weight, max_weight) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

        if sector_caps and sectors and len(sectors) == n:
            sector_caps_clean = {str(k): float(v) for k, v in sector_caps.items() if v is not None}
            for sname, cap in sector_caps_clean.items():
                cap = float(cap)
                if cap <= 0 or cap > 1:
                    raise ValueError(f"Sector cap for {sname} must be in (0, 1].")
                idx = [i for i, sec in enumerate(sectors) if sec == sname]
                if not idx:
                    continue
                constraints.append({
                    "type": "ineq",
                    "fun": (lambda x, idx=idx, cap=cap: cap - float(np.sum(np.asarray(x)[idx])))
                })

        res = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 4000},
        )

        if res.success and np.all(np.isfinite(res.x)):
            w = np.asarray(res.x, dtype=float)
            w = np.clip(w, min_weight, max_weight)
            w = w / (w.sum() if w.sum() != 0 else 1.0)
            return w

        return x0

def _aggregate_horizon_returns(r: pd.Series, horizon_days: int) -> pd.Series:
    if horizon_days <= 1:
        return r.dropna()
    lr = np.log1p(r.dropna())
    agg_lr = lr.rolling(horizon_days).sum().dropna()
    return np.expm1(agg_lr)

def var_cvar_es(
    r: pd.Series,
    conf_levels: Tuple[float, ...] = (0.95, 0.99),
    horizon_days: int = 1,
    methods: Tuple[str, ...] = ("historical", "parametric", "modified"),
) -> pd.DataFrame:
    r_h = _aggregate_horizon_returns(r, horizon_days).replace([np.inf, -np.inf], np.nan).dropna()
    if r_h.empty:
        raise ValueError("Return series is empty after cleaning.")

    mu = float(r_h.mean())
    sigma = float(r_h.std(ddof=1))
    s = float(stats.skew(r_h, bias=False)) if len(r_h) > 10 else 0.0
    k = float(stats.kurtosis(r_h, fisher=False, bias=False)) if len(r_h) > 10 else 3.0

    rows = []
    for cl in conf_levels:
        alpha = 1.0 - float(cl)
        if alpha <= 0 or alpha >= 1:
            continue

        if "historical" in methods:
            q = float(r_h.quantile(alpha))
            var_h = -q
            es_h = -float(r_h[r_h <= q].mean())
            rows.append({"Method": "Historical", "CL": cl, "HorizonDays": horizon_days, "VaR": var_h, "ES(CVaR)": es_h})

        if "parametric" in methods:
            z = float(stats.norm.ppf(alpha))
            qn = mu + sigma * z
            var_p = -qn
            es_p = -(mu - sigma * (stats.norm.pdf(z) / alpha))
            rows.append({"Method": "Parametric(N)", "CL": cl, "HorizonDays": horizon_days, "VaR": var_p, "ES(CVaR)": es_p})

        if "modified" in methods:
            z = float(stats.norm.ppf(alpha))
            z_cf = (
                z
                + (1.0 / 6.0) * (z**2 - 1.0) * s
                + (1.0 / 24.0) * (z**3 - 3.0 * z) * (k - 3.0)
                - (1.0 / 36.0) * (2.0 * z**3 - 5.0 * z) * (s**2)
            )
            qcf = mu + sigma * z_cf
            var_m = -qcf
            es_m = -(mu - sigma * (stats.norm.pdf(z_cf) / alpha))
            rows.append({"Method": "Modified(CF)", "CL": cl, "HorizonDays": horizon_days, "VaR": var_m, "ES(CVaR)": es_m})

    out = pd.DataFrame(rows)
    return out.sort_values(["Method", "CL"]).reset_index(drop=True)

def rolling_risk_contributions_pct(
    returns: pd.DataFrame,
    weights: np.ndarray,
    window: int = 63,
    step: int = 5,
    annualize: float = 252.0,
) -> pd.DataFrame:
    if window < 20:
        raise ValueError("window too small; use at least 20.")
    if step < 1:
        step = 1

    cols = list(returns.columns)
    n = len(cols)

    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.clip(w, 0, None)
    w = w / (w.sum() if w.sum() != 0 else 1.0)
    if w.shape[0] != n:
        raise ValueError("weights length must match number of assets.")

    idx = returns.index
    out_dates, out_rc = [], []

    for end_i in range(window - 1, len(idx), step):
        sub_df = returns.iloc[end_i - window + 1 : end_i + 1].dropna(how="any")
        if len(sub_df) < max(20, window // 3):
            continue

        cov = sub_df.cov().values * annualize
        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(max(port_var, 0.0)))
        if port_vol <= 0:
            rc_pct = np.zeros(n)
        else:
            mrc = (cov @ w) / port_vol
            crc = w * mrc
            rc_pct = (crc / port_vol) * 100.0

        out_dates.append(idx[end_i])
        out_rc.append(rc_pct)

    if not out_rc:
        return pd.DataFrame()

    rc = pd.DataFrame(np.vstack(out_rc), index=pd.DatetimeIndex(out_dates), columns=cols)
    rc_full = rc.reindex(pd.DatetimeIndex(idx)).ffill().dropna(how="all")
    return rc_full

def fig_risk_contribution_bar(risk_metrics: pd.DataFrame) -> go.Figure:
    df = risk_metrics.sort_values("Risk_Contribution_%", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["Company"],
        x=df["Risk_Contribution_%"],
        orientation="h",
        marker=dict(color=df["Risk_Contribution_%"], colorscale="RdYlGn_r", showscale=True,
                    colorbar=dict(title="Risk %")),
        text=df["Risk_Contribution_%"].round(1).astype(str) + "%",
        textposition="outside",
        name="Risk Contribution",
    ))
    equal_target = 100.0 / len(df)
    fig.add_vline(x=equal_target, line_dash="dash", line_color="red", opacity=0.7,
                  annotation_text=f"Equal Risk Target ({equal_target:.1f}%)")
    fig.update_layout(
        title="Risk Contribution by Asset (Ranked)",
        xaxis_title="Risk Contribution (%)",
        yaxis_title="",
        height=720,
        showlegend=False,
        hovermode="y",
        margin=dict(l=10, r=10, t=60, b=20),
    )
    return fig

def fig_risk_concentration_pie(risk_metrics: pd.DataFrame) -> go.Figure:
    df = risk_metrics.sort_values("Risk_Contribution_%", ascending=True)
    top_3 = float(df.tail(3)["Risk_Contribution_%"].sum())
    top_5 = float(df.tail(5)["Risk_Contribution_%"].sum())
    others = max(0.0, 100.0 - top_5)
    fig = go.Figure(data=[go.Pie(labels=["Top 3", "Next 2", "Remaining"],
                                 values=[top_3, top_5 - top_3, others], hole=0.4)])
    fig.update_layout(
        title="Risk Concentration Analysis",
        annotations=[dict(text=f"Top 3: {top_3:.1f}%", x=0.5, y=0.5, font_size=14, showarrow=False)],
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

def fig_sector_risk_vs_weight(sector_analysis: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Risk Contribution", x=sector_analysis.index, y=sector_analysis["Total Risk %"]))
    fig.add_trace(go.Bar(name="Portfolio Weight", x=sector_analysis.index, y=sector_analysis["Total Weight %"]))
    fig.update_layout(
        title="Sector Risk vs Weight Allocation",
        barmode="group",
        height=460,
        xaxis_tickangle=-45,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

def fig_var_table(var_df: pd.DataFrame) -> go.Figure:
    df = var_df.copy()
    df["VaR_%"] = df["VaR"] * 100.0
    df["ES_%"] = df["ES(CVaR)"] * 100.0
    df["CL"] = df["CL"].map(lambda x: f"{int(round(x*100))}%")
    df["HorizonDays"] = df["HorizonDays"].astype(int)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(["Method", "CL", "HorizonDays", "VaR (%)", "ES/CVaR (%)"])),
        cells=dict(values=[
            df["Method"].tolist(),
            df["CL"].tolist(),
            df["HorizonDays"].tolist(),
            [f"{x:.2f}" for x in df["VaR_%"].tolist()],
            [f"{x:.2f}" for x in df["ES_%"].tolist()],
        ])
    )])
    fig.update_layout(title="Portfolio VaR / ES (loss magnitude, %)", height=320, margin=dict(l=10, r=10, t=60, b=10))
    return fig

def fig_rolling_top_contributors(rc_pct: pd.DataFrame, name_map: Dict[str, str], top_n: int = 10) -> go.Figure:
    if rc_pct is None or rc_pct.empty:
        return go.Figure()
    avg = rc_pct.mean(axis=0).sort_values(ascending=False)
    top_cols = avg.head(top_n).index.tolist()
    fig = go.Figure()
    for c in top_cols:
        fig.add_trace(go.Scatter(x=rc_pct.index, y=rc_pct[c], mode="lines", name=name_map.get(c, c)))
    fig.update_layout(
        title=f"Rolling Risk Contribution (%) — Top {top_n}",
        height=520,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=str(name)[:31], index=False)
    bio.seek(0)
    return bio.read()

def main():
    st.markdown('<div class="main-header">📊 BIST Risk Budgeting Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge">📡 Data Source: Yahoo Finance (yfinance)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">If your app fails to build on Community Cloud, set the Python version in the Cloud UI (not runtime.txt).</div>',
        unsafe_allow_html=True,
    )

    analyzer = BISTRiskAnalyzer()
    ok, msg = analyzer.yahoo_health_check()

    with st.sidebar:
        st.markdown("## ⚙️ Parameters")
        st.markdown(f"**Yahoo Health:** {'✅' if ok else '⚠️'}")
        st.caption(msg)

        universe_mode = st.selectbox("Universe", ["Default 20 (stable)", "AUTO_BIST50 (scrape)", "Custom list"])
        custom_text = ""
        if universe_mode == "Custom list":
            custom_text = st.text_area(
                "Tickers (comma/newline). You can write without .IS (we auto-append).",
                value="AKBNK,ARCLK,ASELS,BIMAS,EKGYO,EREGL,FROTO,GARAN,HALKB,ISCTR,KCHOL,KOZAL,KRDMD,PETKM,PGSUS,SAHOL,SASA,TCELL,THYAO,TOASO",
                height=120,
            )

        today = date.today()
        start_date = st.date_input("Start date", value=date(2022, 1, 1))
        end_date = st.date_input("End date", value=today)
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()

        min_obs = st.slider("Min observations per ticker", 30, 260, 90, step=10)
        max_retries = st.slider("Yahoo retries", 1, 6, 3, step=1)

        st.divider()

        portfolio_type = st.radio("Portfolio type", ["Equal Weight", "Risk Parity (Optimized)"])

        st.markdown("### Constraints")
        min_weight = st.slider("Min weight per stock", 0.0, 0.05, 0.0, step=0.005)
        max_weight = st.slider("Max weight per stock", 0.02, 0.30, 0.10, step=0.01)

        st.markdown("### Sector caps (optional)")
        st.caption("Upload a CSV with columns: ticker, sector. Then provide caps as JSON like: {\"Banking\":0.25}.")
        sector_file = st.file_uploader("Sector map CSV", type=["csv"])
        sector_caps_json = st.text_area("Sector caps JSON", value="", height=80)

        st.divider()

        st.markdown("### VaR / ES (CVaR)")
        var_horizon = st.slider("Horizon (days)", 1, 20, 1, step=1)
        cl_sel = st.multiselect("Confidence levels", options=[0.90, 0.95, 0.99], default=[0.95, 0.99])
        methods_sel = st.multiselect(
            "Methods",
            options=["historical", "parametric", "modified"],
            default=["historical", "parametric", "modified"],
        )

        st.divider()

        st.markdown("### Rolling risk contributions")
        compute_rolling = st.checkbox("Compute rolling RC", value=True)
        roll_window = st.slider("Rolling window", 20, 260, 63, step=5)
        roll_step = st.slider("Rolling step", 1, 20, 5, step=1)
        roll_topn = st.slider("Top N assets", 3, 25, 10, step=1)

        st.divider()
        run_btn = st.button("🚀 Run analysis", use_container_width=True)

    if not run_btn:
        st.info("Set parameters on the left, then click **Run analysis**.")
        return

    if universe_mode.startswith("Default"):
        universe = analyzer.default_20
        note = "Default 20"
    elif universe_mode.startswith("AUTO"):
        with st.spinner("Scraping BIST50 constituents..."):
            try:
                universe = analyzer.fetch_bist50_tickers_online()
                note = "AUTO_BIST50 (scraped)"
            except Exception as e:
                st.error(f"Failed to auto-fetch BIST50 list: {e}")
                st.stop()
    else:
        t_in = parse_ticker_input(custom_text)
        universe = analyzer.default_20 if t_in is None else (["AUTO_BIST50"] if t_in == ["AUTO_BIST50"] else t_in)
        note = "Custom"
        if universe == ["AUTO_BIST50"]:
            with st.spinner("Scraping BIST50 constituents..."):
                universe = analyzer.fetch_bist50_tickers_online()
                note = "AUTO_BIST50 (scraped)"

    universe, mapping_used = apply_ticker_aliases(universe)

    sector_map: Optional[Dict[str, str]] = None
    if sector_file is not None:
        try:
            df_sec = pd.read_csv(sector_file)
            c1 = c2 = None
            for c in df_sec.columns:
                lc = str(c).strip().lower()
                if lc in ("ticker", "symbol", "sembol", "kod"):
                    c1 = c
                if lc in ("sector", "sektor", "industry"):
                    c2 = c
            if c1 is None or c2 is None:
                raise ValueError("CSV must contain ticker and sector columns.")
            sector_map = {}
            for _, row in df_sec[[c1, c2]].dropna().iterrows():
                tk = normalize_tickers([row[c1]])[0]
                sector_map[tk] = str(row[c2])
        except Exception as e:
            st.error(f"Sector CSV parse error: {e}")
            st.stop()

    sector_caps: Optional[Dict[str, float]] = None
    if sector_caps_json.strip():
        try:
            sector_caps = json.loads(sector_caps_json)
            if not isinstance(sector_caps, dict):
                raise ValueError("Sector caps must be a JSON object/dict.")
            sector_caps = {str(k): float(v) for k, v in sector_caps.items()}
        except Exception as e:
            st.error(f"Sector caps JSON error: {e}")
            st.stop()

    if sector_map:
        for k, sec in sector_map.items():
            if k in analyzer.meta:
                analyzer.meta[k] = AssetMeta(analyzer.meta[k].name, str(sec))
            else:
                analyzer.meta[k] = AssetMeta(k, str(sec))

    st.markdown('<div class="sub-header">📥 Data download</div>', unsafe_allow_html=True)
    st.caption(f"Universe: {note} • Requested tickers: {len(universe)}")
    if mapping_used:
        st.caption(f"Ticker alias mapping applied: {mapping_used}")

    with st.spinner("Downloading from Yahoo Finance (this can take ~10–30 seconds)..."):
        prices, returns, err, dropped, diag = analyzer.fetch_yahoo_data_cached(
            tuple(universe), str(start_date), str(end_date), int(min_obs), int(max_retries)
        )

    if diag is not None and not diag.empty:
        st.markdown('<div class="sub-header">🔎 Diagnostics</div>', unsafe_allow_html=True)
        st.dataframe(diag.sort_values("ObsCount", ascending=False), use_container_width=True, height=280)

    if err:
        st.error(f"❌ Data error: {err}")
        if dropped:
            st.warning(f"Yahoo returned no data for ({len(dropped)}): {dropped}")
        st.info("Try: (1) Start date later (e.g., 2022-01-01), (2) lower min_obs (e.g., 60), (3) remove problematic tickers.")
        st.stop()

    if dropped:
        st.warning(f"⚠️ Dropped/no-data tickers ({len(dropped)}): {dropped}")

    st.success(f"✅ Loaded {returns.shape[1]} tickers • {len(returns)} trading days")

    loaded = list(returns.columns)
    sectors = []
    for t in loaded:
        if sector_map and t in sector_map:
            sectors.append(str(sector_map[t]))
        else:
            sectors.append(analyzer.meta.get(t, AssetMeta(t, "Other")).sector)

    risk_metrics_eq, portfolio_metrics_eq, cov_matrix = analyzer.calculate_risk_metrics(returns, weights=None)

    if portfolio_type == "Equal Weight":
        w = np.ones(len(loaded)) / len(loaded)
        risk_metrics, portfolio_metrics, _ = analyzer.calculate_risk_metrics(returns, weights=w)

        rp_w = analyzer.calculate_risk_parity_constrained(
            cov_matrix=cov_matrix,
            tickers=loaded,
            sectors=sectors,
            max_weight=float(max_weight),
            min_weight=float(min_weight),
            sector_caps=sector_caps,
        )
    else:
        rp_w = analyzer.calculate_risk_parity_constrained(
            cov_matrix=cov_matrix,
            tickers=loaded,
            sectors=sectors,
            max_weight=float(max_weight),
            min_weight=float(min_weight),
            sector_caps=sector_caps,
        )
        w = rp_w
        risk_metrics, portfolio_metrics, _ = analyzer.calculate_risk_metrics(returns, weights=w)

    sector_analysis = (
        risk_metrics.groupby("Sector", as_index=True)
        .agg({"Risk_Contribution_%": "sum", "Weight": "sum"})
        .rename(columns={"Risk_Contribution_%": "Total Risk %", "Weight": "Total Weight"})
    )
    sector_analysis["Total Weight %"] = sector_analysis["Total Weight"] * 100.0
    sector_analysis = sector_analysis.drop(columns=["Total Weight"]).sort_values("Total Risk %", ascending=False)

    pr = portfolio_returns(returns.fillna(0.0), w)
    var_df = var_cvar_es(pr, conf_levels=tuple(cl_sel), horizon_days=int(var_horizon), methods=tuple(methods_sel))

    rolling_rc = pd.DataFrame()
    if compute_rolling:
        with st.spinner("Computing rolling risk contributions..."):
            rolling_rc = rolling_risk_contributions_pct(
                returns=returns, weights=w, window=int(roll_window), step=int(roll_step), annualize=252.0
            )

    st.markdown('<div class="sub-header">📌 Key portfolio metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualized Volatility", f"{portfolio_metrics['volatility']:.2%}")
    c2.metric("Avg Individual Vol", f"{portfolio_metrics['avg_volatility']:.2%}")
    c3.metric("Diversification Ratio", f"{portfolio_metrics['diversification_ratio']:.2f}")
    c4.metric("Assets Loaded", f"{portfolio_metrics['n_assets']}")

    st.markdown('<div class="sub-header">🎯 Risk contribution analysis</div>', unsafe_allow_html=True)
    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(fig_risk_contribution_bar(risk_metrics), use_container_width=True)
    with right:
        st.plotly_chart(fig_risk_concentration_pie(risk_metrics), use_container_width=True)

    st.markdown('<div class="sub-header">🏭 Sector risk</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_sector_risk_vs_weight(sector_analysis), use_container_width=True)

    st.markdown('<div class="sub-header">📉 VaR / ES</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_var_table(var_df), use_container_width=True)

    if compute_rolling and rolling_rc is not None and not rolling_rc.empty:
        st.markdown('<div class="sub-header">🧷 Rolling risk contributions</div>', unsafe_allow_html=True)
        name_map = {c: analyzer.meta.get(c, AssetMeta(c, "Other")).name for c in rolling_rc.columns}
        st.plotly_chart(fig_rolling_top_contributors(rolling_rc, name_map, top_n=int(roll_topn)), use_container_width=True)
    elif compute_rolling:
        st.warning("Rolling RC produced no output (window may be too large for available data).")

    st.markdown('<div class="sub-header">📋 Detailed risk metrics</div>', unsafe_allow_html=True)
    show_cols = ["Risk_Rank", "Symbol", "Company", "Sector", "Weight", "Risk_Contribution_%", "Individual_Volatility", "Beta"]
    view = risk_metrics[show_cols].copy()
    view["Weight"] = (view["Weight"] * 100).round(3)
    view["Risk_Contribution_%"] = view["Risk_Contribution_%"].round(3)
    view["Individual_Volatility"] = (view["Individual_Volatility"] * 100).round(3)
    view = view.rename(columns={"Weight": "Weight (%)", "Risk_Contribution_%": "Risk Contribution (%)", "Individual_Volatility": "Indiv Vol (%)"})
    st.dataframe(view, use_container_width=True, height=520)

    st.download_button(
        "⬇️ Download risk metrics (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"bist_risk_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    st.markdown('<div class="sub-header">⚖️ Risk parity weights</div>', unsafe_allow_html=True)
    w_df = pd.DataFrame({
        "Symbol": loaded,
        "Company": [analyzer.meta.get(s, AssetMeta(s, "Other")).name for s in loaded],
        "Sector": [analyzer.meta.get(s, AssetMeta(s, "Other")).sector for s in loaded],
        "EqualWeight(%)": np.round(100.0 / len(loaded), 3),
        "RiskParity(%)": np.round(100.0 * rp_w, 3),
        "Delta(%)": np.round(100.0 * (rp_w - (np.ones(len(loaded)) / len(loaded))), 3),
    }).sort_values("RiskParity(%)", ascending=False)

    st.dataframe(w_df, use_container_width=True, height=420)
    st.download_button(
        "⬇️ Download weights comparison (CSV)",
        data=w_df.to_csv(index=False).encode("utf-8"),
        file_name=f"bist_weights_compare_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    st.markdown('<div class="sub-header">📦 Full report</div>', unsafe_allow_html=True)
    sheets = {
        "RiskMetrics": view,
        "WeightsCompare": w_df,
        "SectorAnalysis": sector_analysis.reset_index(),
        "VaR_ES": var_df,
    }
    if compute_rolling and rolling_rc is not None and not rolling_rc.empty:
        rc_export = rolling_rc.reset_index().rename(columns={"index": "Date"})
        sheets["RollingRC"] = rc_export

    xlsx_bytes = to_excel_bytes(sheets)
    st.download_button(
        "⬇️ Download full report (Excel)",
        data=xlsx_bytes,
        file_name=f"bist_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

if __name__ == "__main__":
    main()
