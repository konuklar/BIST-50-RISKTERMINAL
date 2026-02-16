import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="BIST 50 Risk Budgeting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .success-box {
        background-color: #DCFCE7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #059669;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #D97706;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #FEE2E2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #DC2626;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class BIST50RiskAnalyzer:
    """Risk Budgeting Analysis for BIST 50 Stocks using Yahoo Finance"""
    
    # CORRECT Yahoo Finance BIST tickers (verified working as of 2024)
    # Format: TICKER.IS is the correct format for BIST stocks
    tickers = [
        'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'EKGYO.IS',
        'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'HALKB.IS', 'ISCTR.IS',
        'KCHOL.IS', 'KOZAL.IS', 'KRDMD.IS', 'PETKM.IS', 'PGSUS.IS',
        'SAHOL.IS', 'SASA.IS', 'TCELL.IS', 'THYAO.IS', 'TOASO.IS'
    ]
    
    # Company names mapping
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
    
    # Sectors mapping
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
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def test_yahoo_connection(_self):
        """Test if Yahoo Finance is accessible"""
        try:
            test_ticker = yf.Ticker('AKBNK.IS')
            test_data = test_ticker.history(period="5d")
            return not test_data.empty
        except:
            return False
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_yahoo_data(_self, start_date, end_date):
        """
        Fetch data from Yahoo Finance with optimized batch downloading
        """
        
        try:
            # Convert dates to string format
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Use batch download for better performance
            status_text.text("Connecting to Yahoo Finance...")
            
            # Download all tickers at once (more efficient)
            data = yf.download(
                tickers=_self.tickers,
                start=start_str,
                end=end_str,
                progress=False,
                group_by='ticker',
                auto_adjust=True,
                timeout=30,
                threads=True
            )
            
            progress_bar.progress(0.5)
            status_text.text("Processing data...")
            
            if data.empty:
                progress_bar.empty()
                status_text.empty()
                return None, None, _self.tickers
            
            # Extract Adjusted Close prices
            try:
                if len(_self.tickers) == 1:
                    prices = pd.DataFrame({_self.tickers[0]: data['Adj Close']})
                else:
                    # Try to get Adj Close from multi-index
                    if isinstance(data.columns, pd.MultiIndex):
                        prices = data.xs('Adj Close', axis=1, level=0)
                    else:
                        # Fallback to Close prices
                        prices = data['Close'] if 'Close' in data else data
            except:
                # Final fallback
                prices = data
            
            progress_bar.progress(0.75)
            
            # Ensure we have a DataFrame
            if isinstance(prices, pd.Series):
                prices = pd.DataFrame(prices)
            
            # Clean data
            prices = prices.sort_index()
            prices = prices.ffill(limit=3)
            prices = prices.dropna()
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Identify successful and failed tickers
            successful_tickers = prices.columns.tolist()
            failed_tickers = [t for t in _self.tickers if t not in successful_tickers]
            
            progress_bar.empty()
            status_text.empty()
            
            if returns.empty:
                return None, None, _self.tickers
            
            return prices, returns, failed_tickers
            
        except Exception as e:
            st.error(f"Yahoo Finance connection error: {str(e)}")
            return None, None, _self.tickers
    
    def calculate_risk_metrics(self, returns):
        """Calculate comprehensive risk metrics for equally weighted portfolio"""
        
        n_assets = len(returns.columns)
        weights = np.ones(n_assets) / n_assets
        
        # Annualized covariance matrix
        cov_matrix = returns.cov() * 252
        
        # Portfolio metrics
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Individual volatilities
        indiv_vol = np.sqrt(np.diag(cov_matrix))
        
        # Marginal Risk Contributions (MRC)
        marginal_risk = (cov_matrix @ weights) / portfolio_volatility
        
        # Component Risk Contributions (CRC)
        component_risk = weights * marginal_risk
        
        # Percentage contributions
        pct_contributions = (component_risk / portfolio_volatility) * 100
        
        # Create DataFrame
        risk_metrics = pd.DataFrame({
            'Symbol': returns.columns,
            'Company': [self.asset_names.get(t, t) for t in returns.columns],
            'Sector': [self.sectors.get(t, 'Other') for t in returns.columns],
            'Weight': weights,
            'Individual_Volatility': indiv_vol,
            'Marginal_Risk_Contribution': marginal_risk,
            'Component_Risk': component_risk,
            'Risk_Contribution_%': pct_contributions
        })
        
        # Sort by risk contribution
        risk_metrics = risk_metrics.sort_values('Risk_Contribution_%', ascending=False)
        risk_metrics['Risk_Rank'] = range(1, len(risk_metrics) + 1)
        
        # Calculate betas
        portfolio_returns = returns @ weights
        betas = []
        for col in returns.columns:
            cov = returns[col].cov(portfolio_returns) * 252
            beta = cov / portfolio_variance
            betas.append(beta)
        
        risk_metrics['Beta'] = betas
        
        # Calculate diversification ratio
        weighted_avg_vol = np.sum(weights * indiv_vol)
        diversification_ratio = weighted_avg_vol / portfolio_volatility
        
        portfolio_metrics = {
            'volatility': portfolio_volatility,
            'diversification_ratio': diversification_ratio,
            'n_assets': n_assets,
            'avg_volatility': indiv_vol.mean(),
            'max_risk_contrib': risk_metrics.iloc[0]['Risk_Contribution_%'],
            'max_risk_asset': risk_metrics.iloc[0]['Company']
        }
        
        return risk_metrics, portfolio_metrics, cov_matrix
    
    def calculate_risk_parity(self, cov_matrix, returns_columns):
        """Calculate risk parity weights"""
        
        n = len(returns_columns)
        
        def risk_parity_objective(weights):
            weights = weights / np.sum(weights)
            portfolio_var = weights @ cov_matrix @ weights
            portfolio_vol = np.sqrt(portfolio_var)
            
            marginal_contrib = (cov_matrix @ weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            
            target_rc = portfolio_vol / n
            rc_deviation = risk_contrib - target_rc
            
            return np.sum(rc_deviation ** 2)
        
        # Initial guess
        init_weights = np.ones(n) / n
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(n)]
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-8, 'maxiter': 1000}
        )
        
        if result.success:
            return result.x / np.sum(result.x)
        else:
            return init_weights

def main():
    # Header
    st.markdown('<p class="main-header">📊 BIST 50 Risk Budgeting Dashboard</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard analyzes **Marginal Risk Contributions (MRC)** and provides **Risk Budgeting** 
    recommendations for an equally weighted portfolio of 20 major BIST 50 stocks.
    """)
    
    # Initialize analyzer
    analyzer = BIST50RiskAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Parameters")
        
        # Date range selection
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
        
        # Ensure dates are valid
        if start_date >= end_date:
            st.error("Start date must be before end date")
            st.stop()
        
        # Portfolio type
        portfolio_type = st.radio(
            "Portfolio Type",
            ["Equal Weight", "Risk Parity (Optimized)"],
            help="Equal weight portfolio or risk parity optimization"
        )
        
        # Display options
        st.markdown("### 📈 Display Options")
        show_individual_vol = st.checkbox("Individual Volatility", value=True)
        show_mrc = st.checkbox("Marginal Risk Contribution (MRC)", value=True)
        show_beta = st.checkbox("Beta Coefficients", value=True)
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Yahoo Finance Status
        if analyzer.test_yahoo_connection():
            st.markdown("✅ **Yahoo Finance:** Connected")
        else:
            st.markdown("⚠️ **Yahoo Finance:** Checking...")
    
    # Main content
    try:
        # Test connection first
        if not analyzer.test_yahoo_connection():
            st.markdown("""
            <div class="warning-box">
                ⚠️ Yahoo Finance connection is unstable. Please wait a moment and try refreshing.
                <br><br>
                <strong>Troubleshooting:</strong>
                <ul>
                    <li>Check your internet connection</li>
                    <li>Try a different date range</li>
                    <li>Refresh the page in 1-2 minutes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Retry Connection"):
                st.cache_data.clear()
                st.rerun()
        
        # Fetch data
        with st.spinner("📥 Fetching data from Yahoo Finance..."):
            prices, returns, failed_tickers = analyzer.fetch_yahoo_data(start_date, end_date)
        
        if prices is not None and returns is not None and not returns.empty:
            
            # Show success message
            st.markdown("""
            <div class="success-box">
                ✅ Successfully connected to Yahoo Finance
            </div>
            """, unsafe_allow_html=True)
            
            # Show failed tickers if any
            if failed_tickers:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ {len(failed_tickers)} out of 20 tickers could not be loaded.
                    Working with {len(returns.columns)} stocks.
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View unavailable tickers"):
                    for ticker in failed_tickers:
                        st.write(f"- {ticker} ({analyzer.asset_names.get(ticker, ticker)})")
            
            # Display data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trading Days", len(returns))
            with col2:
                st.metric("Stocks Analyzed", len(returns.columns))
            with col3:
                period_days = (end_date - start_date).days
                st.metric("Analysis Period", f"{period_days} days")
            
            # Calculate metrics
            risk_metrics, portfolio_metrics, cov_matrix = analyzer.calculate_risk_metrics(returns)
            
            # Key Metrics Dashboard
            st.markdown('<p class="sub-header">📌 Key Portfolio Metrics</p>', 
                       unsafe_allow_html=True)
            
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            
            with mcol1:
                st.metric(
                    "Portfolio Volatility",
                    f"{portfolio_metrics['volatility']:.2%}",
                    help="Annualized portfolio volatility"
                )
            
            with mcol2:
                st.metric(
                    "Avg Individual Vol",
                    f"{portfolio_metrics['avg_volatility']:.2%}",
                    help="Average volatility of individual stocks"
                )
            
            with mcol3:
                div_ratio = portfolio_metrics['diversification_ratio']
                st.metric(
                    "Diversification Ratio",
                    f"{div_ratio:.2f}",
                    delta="Good" if div_ratio > 1.5 else "Low",
                    delta_color="normal" if div_ratio > 1.5 else "inverse",
                    help=">1.5 indicates good diversification"
                )
            
            with mcol4:
                st.metric(
                    "Top Risk Contributor",
                    portfolio_metrics['max_risk_asset'],
                    f"{portfolio_metrics['max_risk_contrib']:.1f}%",
                    help="Stock with highest risk contribution"
                )
            
            # Risk Contribution Analysis
            st.markdown('<p class="sub-header">🎯 Risk Contribution Analysis</p>', 
                       unsafe_allow_html=True)
            
            # Horizontal bar chart of risk contributions
            fig = go.Figure()
            
            sorted_df = risk_metrics.sort_values('Risk_Contribution_%', ascending=True)
            
            # Color based on contribution relative to target
            equal_contrib = 100 / len(sorted_df)
            colors = []
            for x in sorted_df['Risk_Contribution_%']:
                if x > equal_contrib * 1.2:
                    colors.append('#DC2626')  # Red for high contribution
                elif x < equal_contrib * 0.8:
                    colors.append('#10B981')  # Green for low contribution
                else:
                    colors.append('#F59E0B')  # Orange for near target
            
            fig.add_trace(go.Bar(
                y=sorted_df['Company'],
                x=sorted_df['Risk_Contribution_%'],
                orientation='h',
                marker_color=colors,
                text=sorted_df['Risk_Contribution_%'].round(1).astype(str) + '%',
                textposition='outside',
                name='Risk Contribution',
                hovertemplate='<b>%{y}</b><br>' +
                              'Risk Contribution: %{x:.1f}%<br>' +
                              'Sector: %{customdata}<br>' +
                              '<extra></extra>',
                customdata=sorted_df['Sector']
            ))
            
            # Equal contribution reference line
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
                hovermode='y',
                xaxis=dict(range=[0, max(sorted_df['Risk_Contribution_%']) * 1.1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Metrics Table
            st.markdown('<p class="sub-header">📋 Detailed Risk Metrics</p>', 
                       unsafe_allow_html=True)
            
            # Prepare display columns
            display_cols = ['Risk_Rank', 'Company', 'Sector', 'Weight', 'Risk_Contribution_%']
            
            if show_individual_vol:
                display_cols.append('Individual_Volatility')
            if show_mrc:
                display_cols.append('Marginal_Risk_Contribution')
            if show_beta:
                display_cols.append('Beta')
            
            display_df = risk_metrics[display_cols].copy()
            
            # Format columns
            display_df['Weight'] = display_df['Weight'].map('{:.1%}'.format)
            display_df['Risk_Contribution_%'] = display_df['Risk_Contribution_%'].map('{:.1f}%'.format)
            
            if show_individual_vol:
                display_df['Individual_Volatility'] = display_df['Individual_Volatility'].map('{:.1%}'.format)
            if show_mrc:
                display_df['Marginal_Risk_Contribution'] = display_df['Marginal_Risk_Contribution'].map('{:.3f}'.format)
            if show_beta:
                display_df['Beta'] = display_df['Beta'].map('{:.2f}'.format)
            
            # Rename columns
            column_names = {
                'Risk_Rank': 'Rank',
                'Company': 'Company',
                'Sector': 'Sector',
                'Weight': 'Weight',
                'Risk_Contribution_%': 'Risk %',
                'Individual_Volatility': 'Indiv Vol',
                'Marginal_Risk_Contribution': 'MRC',
                'Beta': 'Beta'
            }
            
            display_df = display_df.rename(columns=column_names)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Risk %": st.column_config.ProgressColumn(
                        "Risk Contribution",
                        help="Percentage of total portfolio risk",
                        format="%s",
                        min_value=0,
                        max_value=100
                    )
                }
            )
            
            # Risk Parity Recommendations (for Equal Weight portfolio)
            if portfolio_type == "Equal Weight" and len(returns.columns) > 1:
                st.markdown('<p class="sub-header">⚖️ Risk Parity Recommendations</p>', 
                           unsafe_allow_html=True)
                
                rp_weights = analyzer.calculate_risk_parity(cov_matrix, returns.columns)
                
                # Create recommendations
                recommendations = []
                for i, symbol in enumerate(returns.columns):
                    current_w = 1/len(returns.columns)
                    rp_w = rp_weights[i]
                    adjustment = rp_w - current_w
                    
                    company = analyzer.asset_names.get(symbol, symbol)
                    sector = analyzer.sectors.get(symbol, 'Other')
                    
                    action = 'REDUCE' if adjustment < -0.005 else 'INCREASE' if adjustment > 0.005 else 'MAINTAIN'
                    
                    recommendations.append({
                        'Company': company,
                        'Sector': sector,
                        'Current Weight': f"{current_w:.1%}",
                        'Risk Parity Weight': f"{rp_w:.1%}",
                        'Adjustment': f"{adjustment:+.1%}",
                        'Action': action
                    })
                
                rec_df = pd.DataFrame(recommendations)
                
                # Display recommendations
                st.dataframe(
                    rec_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv = rec_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Recommendations (CSV)",
                    data=csv,
                    file_name=f"risk_parity_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            # Export option
            if st.button("📊 Generate Excel Report"):
                output = pd.ExcelWriter('risk_report.xlsx', engine='xlsxwriter')
                
                risk_metrics.to_excel(output, sheet_name='Risk Metrics', index=False)
                
                if portfolio_type == "Equal Weight" and len(returns.columns) > 1:
                    rec_df.to_excel(output, sheet_name='Recommendations', index=False)
                
                # Add summary sheet
                summary = pd.DataFrame({
                    'Metric': ['Portfolio Volatility', 'Diversification Ratio', 'Analysis Period'],
                    'Value': [
                        f"{portfolio_metrics['volatility']:.2%}",
                        f"{portfolio_metrics['diversification_ratio']:.2f}",
                        f"{start_date} to {end_date}"
                    ]
                })
                summary.to_excel(output, sheet_name='Summary', index=False)
                
                output.close()
                
                with open('risk_report.xlsx', 'rb') as f:
                    st.download_button(
                        label="📥 Download Full Report (Excel)",
                        data=f,
                        file_name=f"bist50_risk_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            ❌ An error occurred: {str(e)}
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Troubleshooting Steps:**
        1. Check your internet connection
        2. Visit finance.yahoo.com to verify it's accessible
        3. Try a different date range (some historical data might be unavailable)
        4. Wait a few minutes and refresh
        5. If problem persists, Yahoo Finance API might be temporarily down
        """)

if __name__ == "__main__":
    main()
