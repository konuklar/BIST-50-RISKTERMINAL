import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta
import io
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

class BIST50RiskAnalyzer:
    """Risk Budgeting Analysis for BIST 50 Stocks using Yahoo Finance"""
    
    # Yahoo Finance BIST tickers
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
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_yahoo_data(_self, start_date, end_date):
        """
        Fetch data from Yahoo Finance with robust price extraction
        """
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Connecting to Yahoo Finance...")
            progress_bar.progress(10)
            
            # Download data
            data = yf.download(
                tickers=_self.tickers,
                start=start_str,
                end=end_str,
                progress=False,
                group_by='ticker',
                auto_adjust=True,
                timeout=30
            )
            
            progress_bar.progress(50)
            status_text.text("Processing data...")
            
            if data is None or data.empty:
                progress_bar.empty()
                status_text.empty()
                return None, None, _self.tickers
            
            # Extract price data correctly
            prices = pd.DataFrame()
            
            if len(_self.tickers) == 1:
                # Single ticker case
                if isinstance(data, pd.DataFrame):
                    # With auto_adjust=True, we get OHLC columns directly
                    prices[_self.tickers[0]] = data['Close'] if 'Close' in data else data.iloc[:, 0]
            else:
                # Multi-ticker case
                for ticker in _self.tickers:
                    if ticker in data.columns.levels[0]:
                        # Get the data for this ticker
                        ticker_data = data[ticker]
                        # Use Close price (with auto_adjust=True, this is adjusted close)
                        if 'Close' in ticker_data.columns:
                            prices[ticker] = ticker_data['Close']
            
            progress_bar.progress(75)
            
            if prices.empty:
                progress_bar.empty()
                status_text.empty()
                return None, None, _self.tickers
            
            # Clean data
            prices = prices.sort_index()
            prices = prices.ffill(limit=3)
            prices = prices.dropna()
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Identify failed tickers
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
    
    def calculate_risk_metrics(self, returns, weights=None):
        """
        Calculate comprehensive risk metrics with custom weights
        If weights is None, uses equal weights
        """
        n_assets = len(returns.columns)
        
        # Use provided weights or default to equal weight
        if weights is None:
            weights = np.ones(n_assets) / n_assets
        else:
            # Ensure weights sum to 1
            weights = np.array(weights) / np.sum(weights)
        
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
        
        # Percentage contributions (as decimals, will be multiplied by 100 for display)
        pct_contributions = (component_risk / portfolio_volatility)  # This gives 0-1 scale
        
        # Create DataFrame
        risk_metrics = pd.DataFrame({
            'Symbol': returns.columns,
            'Company': [self.asset_names.get(t, t) for t in returns.columns],
            'Sector': [self.sectors.get(t, 'Other') for t in returns.columns],
            'Weight': weights,
            'Individual_Volatility': indiv_vol,
            'Marginal_Risk_Contribution': marginal_risk,
            'Component_Risk': component_risk,
            'Risk_Contribution': pct_contributions  # Store as decimal for calculations
        })
        
        # Sort by risk contribution
        risk_metrics = risk_metrics.sort_values('Risk_Contribution', ascending=False)
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
            'max_risk_contrib': risk_metrics.iloc[0]['Risk_Contribution'] * 100,  # Convert to %
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
    
    def create_excel_report(self, risk_metrics, recommendations, portfolio_metrics, 
                           start_date, end_date, portfolio_type):
        """Create Excel report in memory"""
        output = io.BytesIO()
        
        try:
            # Try using xlsxwriter first
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                risk_metrics_display = risk_metrics.copy()
                risk_metrics_display['Weight'] = risk_metrics_display['Weight'].map('{:.1%}'.format)
                risk_metrics_display['Risk_Contribution'] = risk_metrics_display['Risk_Contribution'].map('{:.1%}'.format)
                risk_metrics_display['Individual_Volatility'] = risk_metrics_display['Individual_Volatility'].map('{:.1%}'.format)
                risk_metrics_display['Marginal_Risk_Contribution'] = risk_metrics_display['Marginal_Risk_Contribution'].map('{:.3f}'.format)
                risk_metrics_display['Beta'] = risk_metrics_display['Beta'].map('{:.2f}'.format)
                
                risk_metrics_display.to_excel(writer, sheet_name='Risk Metrics', index=False)
                
                if recommendations is not None:
                    recommendations.to_excel(writer, sheet_name='Recommendations', index=False)
                
                # Summary sheet
                summary = pd.DataFrame({
                    'Metric': ['Portfolio Volatility', 'Diversification Ratio', 
                              'Analysis Period', 'Portfolio Type', 'Number of Stocks'],
                    'Value': [
                        f"{portfolio_metrics['volatility']:.2%}",
                        f"{portfolio_metrics['diversification_ratio']:.2f}",
                        f"{start_date} to {end_date}",
                        portfolio_type,
                        len(risk_metrics)
                    ]
                })
                summary.to_excel(writer, sheet_name='Summary', index=False)
        except:
            # Fallback to openpyxl
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                risk_metrics.to_excel(writer, sheet_name='Risk Metrics', index=False)
                if recommendations is not None:
                    recommendations.to_excel(writer, sheet_name='Recommendations', index=False)
        
        output.seek(0)
        return output

def main():
    # Header
    st.markdown('<p class="main-header">📊 BIST 50 Risk Budgeting Dashboard</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard analyzes **Marginal Risk Contributions (MRC)** and provides **Risk Budgeting** 
    recommendations for a portfolio of 20 major BIST 50 stocks.
    """)
    
    # Data source badge
    st.markdown("""
    <div class="data-source-badge">
        📡 Data Source: Yahoo Finance (Real-time)
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = BIST50RiskAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Borsa_Istanbul_logo.svg/200px-Borsa_Istanbul_logo.svg.png", 
                 width=150)
        
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
        show_mrc = st.checkbox("Marginal Risk Contribution", value=True)
        show_beta = st.checkbox("Beta Coefficients", value=True)
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Main content
    try:
        # Fetch data
        with st.spinner("📥 Fetching real-time data from Yahoo Finance..."):
            prices, returns, failed_tickers = analyzer.fetch_yahoo_data(start_date, end_date)
        
        if prices is not None and returns is not None and not returns.empty:
            
            st.markdown("""
            <div class="success-box">
                ✅ Successfully connected to Yahoo Finance
            </div>
            """, unsafe_allow_html=True)
            
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
            
            # Calculate weights based on portfolio type
            if portfolio_type == "Equal Weight":
                weights = None  # Will use equal weights in calculate_risk_metrics
                weights_display = "Equal Weight"
            else:
                # Calculate risk parity weights first
                cov_matrix = returns.cov() * 252
                rp_weights = analyzer.calculate_risk_parity(cov_matrix, returns.columns)
                weights = rp_weights
                weights_display = "Risk Parity"
            
            # Calculate metrics with the correct weights
            risk_metrics, portfolio_metrics, cov_matrix = analyzer.calculate_risk_metrics(returns, weights)
            
            # Display data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trading Days", len(returns))
            with col2:
                st.metric("Stocks Loaded", len(returns.columns))
            with col3:
                st.metric("Portfolio Type", weights_display)
            
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
                    f"{portfolio_metrics['avg_volatility']:.2%}"
                )
            
            with mcol3:
                div_ratio = portfolio_metrics['diversification_ratio']
                st.metric(
                    "Diversification Ratio",
                    f"{div_ratio:.2f}",
                    delta="Good" if div_ratio > 1.5 else "Low",
                    delta_color="normal" if div_ratio > 1.5 else "inverse"
                )
            
            with mcol4:
                st.metric(
                    "Top Risk Contributor",
                    portfolio_metrics['max_risk_asset'],
                    f"{portfolio_metrics['max_risk_contrib']:.1f}%"
                )
            
            # Risk Contribution Analysis
            st.markdown('<p class="sub-header">🎯 Risk Contribution Analysis</p>', 
                       unsafe_allow_html=True)
            
            # Create risk contribution chart
            fig = go.Figure()
            
            sorted_df = risk_metrics.sort_values('Risk_Contribution', ascending=True)
            equal_contrib = 1 / len(sorted_df)  # Target as decimal
            
            colors = []
            for x in sorted_df['Risk_Contribution']:
                if x > equal_contrib * 1.2:
                    colors.append('#DC2626')
                elif x < equal_contrib * 0.8:
                    colors.append('#10B981')
                else:
                    colors.append('#F59E0B')
            
            fig.add_trace(go.Bar(
                y=sorted_df['Company'],
                x=sorted_df['Risk_Contribution'] * 100,  # Convert to % for display
                orientation='h',
                marker_color=colors,
                text=sorted_df['Risk_Contribution'].map('{:.1%}'.format),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>' +
                              'Risk Contribution: %{x:.1f}%<br>' +
                              'Sector: %{customdata}<br>' +
                              '<extra></extra>',
                customdata=sorted_df['Sector']
            ))
            
            fig.add_vline(
                x=equal_contrib * 100,
                line_dash="dash",
                line_color="red",
                opacity=0.7,
                annotation_text=f"Target ({equal_contrib*100:.1f}%)"
            )
            
            fig.update_layout(
                title=f"Risk Contribution by Asset - {weights_display} Portfolio",
                xaxis_title="Risk Contribution (%)",
                yaxis_title="",
                height=600,
                showlegend=False,
                hovermode='y',
                xaxis=dict(range=[0, max(sorted_df['Risk_Contribution'] * 100) * 1.1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Metrics Table
            st.markdown('<p class="sub-header">📋 Detailed Risk Metrics</p>', 
                       unsafe_allow_html=True)
            
            # Prepare display DataFrame with proper numeric values
            display_df = risk_metrics[['Risk_Rank', 'Company', 'Sector', 'Weight', 
                                      'Risk_Contribution']].copy()
            
            if show_individual_vol:
                display_df['Individual_Volatility'] = risk_metrics['Individual_Volatility']
            if show_mrc:
                display_df['Marginal_Risk_Contribution'] = risk_metrics['Marginal_Risk_Contribution']
            if show_beta:
                display_df['Beta'] = risk_metrics['Beta']
            
            # Rename columns
            column_names = {
                'Risk_Rank': 'Rank',
                'Company': 'Company',
                'Sector': 'Sector',
                'Weight': 'Weight',
                'Risk_Contribution': 'Risk %',
                'Individual_Volatility': 'Indiv Vol',
                'Marginal_Risk_Contribution': 'MRC',
                'Beta': 'Beta'
            }
            display_df = display_df.rename(columns=column_names)
            
            # Configure column display
            column_config = {
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Company": st.column_config.TextColumn("Company", width="medium"),
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                "Weight": st.column_config.ProgressColumn(
                    "Weight",
                    help="Portfolio weight",
                    format="%.1f%%",
                    min_value=0,
                    max_value=1
                ),
                "Risk %": st.column_config.ProgressColumn(
                    "Risk %",
                    help="Percentage of total portfolio risk",
                    format="%.1f%%",
                    min_value=0,
                    max_value=1
                )
            }
            
            if show_individual_vol:
                column_config["Indiv Vol"] = st.column_config.NumberColumn(
                    "Indiv Vol",
                    help="Individual stock volatility",
                    format="%.1f%%"
                )
            if show_mrc:
                column_config["MRC"] = st.column_config.NumberColumn(
                    "MRC",
                    help="Marginal Risk Contribution",
                    format="%.3f"
                )
            if show_beta:
                column_config["Beta"] = st.column_config.NumberColumn(
                    "Beta",
                    help="Beta to portfolio",
                    format="%.2f"
                )
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config
            )
            
            # Sector Analysis with correct percentages
            if len(returns.columns) > 1:
                st.markdown('<p class="sub-header">🏭 Sector Risk Analysis</p>', 
                           unsafe_allow_html=True)
                
                sector_analysis = risk_metrics.groupby('Sector').agg({
                    'Risk_Contribution': 'sum',
                    'Weight': 'sum'
                }).round(4)
                
                sector_analysis.columns = ['Total Risk %', 'Total Weight %']
                # Convert to percentages by multiplying by 100
                sector_analysis['Total Risk %'] = sector_analysis['Total Risk %'] * 100
                sector_analysis['Total Weight %'] = sector_analysis['Total Weight %'] * 100
                sector_analysis = sector_analysis.sort_values('Total Risk %', ascending=False)
                
                # Display as DataFrame with proper formatting
                st.dataframe(
                    sector_analysis.style.format({
                        'Total Risk %': '{:.1f}%',
                        'Total Weight %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            
            # Export functionality
            if st.button("📊 Generate Excel Report"):
                # Prepare recommendations if applicable
                recommendations = None
                if portfolio_type == "Equal Weight" and len(returns.columns) > 1:
                    rp_weights = analyzer.calculate_risk_parity(cov_matrix, returns.columns)
                    recs = []
                    for i, symbol in enumerate(returns.columns):
                        current_w = 1/len(returns.columns)
                        rp_w = rp_weights[i]
                        adjustment = rp_w - current_w
                        
                        recs.append({
                            'Company': analyzer.asset_names.get(symbol, symbol),
                            'Current Weight': f"{current_w:.1%}",
                            'Risk Parity Weight': f"{rp_w:.1%}",
                            'Adjustment': f"{adjustment:+.1%}"
                        })
                    recommendations = pd.DataFrame(recs)
                
                # Create Excel file in memory
                excel_file = analyzer.create_excel_report(
                    risk_metrics, recommendations, portfolio_metrics,
                    start_date, end_date, weights_display
                )
                
                st.download_button(
                    label="📥 Download Full Report (Excel)",
                    data=excel_file,
                    file_name=f"bist50_risk_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try refreshing the page or adjusting the date range.")

if __name__ == "__main__":
    main()
