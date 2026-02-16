import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_yahoo_data(_self, start_date, end_date):
        """
        Fetch data directly from Yahoo Finance with proper error handling
        Uses batch download for efficiency
        """
        
        try:
            # Convert dates to string format
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Progress indicators
            progress_placeholder = st.empty()
            progress_placeholder.info("🔄 Connecting to Yahoo Finance...")
            
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
            
            progress_placeholder.empty()
            
            if data.empty:
                st.error("No data received from Yahoo Finance. Please check your internet connection.")
                return None, None
            
            # Extract Adjusted Close prices
            if len(_self.tickers) == 1:
                # Single ticker case
                prices = pd.DataFrame({_self.tickers[0]: data['Adj Close']})
            else:
                # Multi-ticker case
                try:
                    # Try to get Adj Close from multi-index
                    prices = data.xs('Adj Close', axis=1, level=0)
                except:
                    try:
                        # Try alternative structure
                        prices = data['Adj Close']
                    except:
                        # Fall back to Close prices
                        if 'Close' in data:
                            prices = data['Close']
                        else:
                            prices = data
            
            # Ensure we have a DataFrame
            if isinstance(prices, pd.Series):
                prices = pd.DataFrame(prices)
            
            # Clean data
            # Remove any columns that are all NaN
            prices = prices.dropna(axis=1, how='all')
            
            # Forward fill missing values (max 3 days for holidays)
            prices = prices.fillna(method='ffill', limit=3)
            
            # Remove any remaining NaN rows
            prices = prices.dropna()
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            if returns.empty:
                st.error("No valid returns data could be calculated.")
                return None, None
            
            return prices, returns
            
        except Exception as e:
            st.error(f"Yahoo Finance connection error: {str(e)}")
            st.info("Please try again in a few moments or contact support if the issue persists.")
            return None, None
    
    def calculate_risk_metrics(self, returns):
        """Calculate comprehensive risk metrics for equally weighted portfolio"""
        
        n_assets = len(returns.columns)
        weights = np.ones(n_assets) / n_assets
        
        # Annualized covariance matrix (252 trading days)
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
        
        st.markdown("---")
        st.markdown(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.markdown("**Status:** 🟢 Active")
    
    # Main content
    try:
        # Fetch data
        with st.spinner("📥 Fetching real-time data from Yahoo Finance..."):
            prices, returns = analyzer.fetch_yahoo_data(start_date, end_date)
        
        if prices is not None and returns is not None:
            
            # Display data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trading Days", len(returns))
            with col2:
                st.metric("Stocks Loaded", len(returns.columns))
            with col3:
                period_days = (end_date - start_date).days
                st.metric("Period (Days)", period_days)
            
            # Calculate metrics based on portfolio type
            if portfolio_type == "Equal Weight":
                risk_metrics, portfolio_metrics, cov_matrix = analyzer.calculate_risk_metrics(returns)
            else:
                # Calculate risk parity weights
                risk_metrics, portfolio_metrics, cov_matrix = analyzer.calculate_risk_metrics(returns)
                rp_weights = analyzer.calculate_risk_parity(cov_matrix, returns.columns)
                
                # Update weights in risk_metrics
                for i, symbol in enumerate(returns.columns):
                    risk_metrics.loc[risk_metrics['Symbol'] == symbol, 'Weight'] = rp_weights[i]
                
                # Recalculate metrics with new weights
                risk_metrics, portfolio_metrics, cov_matrix = analyzer.calculate_risk_metrics(returns)
            
            # Key Metrics Dashboard
            st.markdown('<p class="sub-header">📌 Key Portfolio Metrics</p>', 
                       unsafe_allow_html=True)
            
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            
            with mcol1:
                st.metric(
                    "Portfolio Volatility (Annual)",
                    f"{portfolio_metrics['volatility']:.2%}"
                )
            
            with mcol2:
                st.metric(
                    "Average Individual Vol",
                    f"{portfolio_metrics['avg_volatility']:.2%}"
                )
            
            with mcol3:
                st.metric(
                    "Diversification Ratio",
                    f"{portfolio_metrics['diversification_ratio']:.2f}",
                    help=">1.5 indicates good diversification"
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
            
            # Create two columns for charts
            chart_col1, chart_col2 = st.columns([2, 1])
            
            with chart_col1:
                # Horizontal bar chart of risk contributions
                fig = go.Figure()
                
                # Sort for better visualization
                sorted_df = risk_metrics.sort_values('Risk_Contribution_%', ascending=True)
                
                fig.add_trace(go.Bar(
                    y=sorted_df['Company'],
                    x=sorted_df['Risk_Contribution_%'],
                    orientation='h',
                    marker=dict(
                        color=sorted_df['Risk_Contribution_%'],
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Risk %")
                    ),
                    text=sorted_df['Risk_Contribution_%'].round(1).astype(str) + '%',
                    textposition='outside',
                    name='Risk Contribution'
                ))
                
                # Equal contribution reference line
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
                    hovermode='y'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_col2:
                # Concentration pie chart
                top_3 = sorted_df.tail(3)['Risk_Contribution_%'].sum()
                top_5 = sorted_df.tail(5)['Risk_Contribution_%'].sum()
                others = 100 - top_5
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Top 3 Contributors', 'Next 2 Contributors', 'Remaining 15'],
                    values=[top_3, top_5 - top_3, others],
                    hole=0.4,
                    marker_colors=['#DC2626', '#F59E0B', '#10B981']
                )])
                
                fig.update_layout(
                    title="Risk Concentration Analysis",
                    annotations=[dict(
                        text=f'Top 3: {top_3:.1f}%',
                        x=0.5, y=0.5,
                        font_size=14,
                        showarrow=False
                    )]
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
            if portfolio_type == "Equal Weight":
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
                    
                    action = 'REDUCE' if adjustment < -0.002 else 'INCREASE' if adjustment > 0.002 else 'MAINTAIN'
                    
                    recommendations.append({
                        'Company': company,
                        'Sector': sector,
                        'Current Weight': f"{current_w:.1%}",
                        'Risk Parity Weight': f"{rp_w:.1%}",
                        'Adjustment': f"{adjustment:+.1%}",
                        'Action': action
                    })
                
                rec_df = pd.DataFrame(recommendations)
                
                # Color coding for actions
                def color_action(val):
                    if val == 'REDUCE':
                        return 'background-color: #FEE2E2; color: #DC2626'
                    elif val == 'INCREASE':
                        return 'background-color: #DCFCE7; color: #059669'
                    else:
                        return 'background-color: #F3F4F6; color: #6B7280'
                
                styled_rec = rec_df.style.applymap(color_action, subset=['Action'])
                
                st.dataframe(
                    styled_rec,
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
            
            # Sector Analysis
            st.markdown('<p class="sub-header">🏭 Sector Risk Analysis</p>', 
                       unsafe_allow_html=True)
            
            sector_analysis = risk_metrics.groupby('Sector').agg({
                'Risk_Contribution_%': 'sum',
                'Weight': 'sum'
            }).round(1)
            
            sector_analysis.columns = ['Total Risk %', 'Total Weight %']
            sector_analysis = sector_analysis.sort_values('Total Risk %', ascending=False)
            
            # Create sector comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Risk Contribution',
                x=sector_analysis.index,
                y=sector_analysis['Total Risk %'],
                marker_color='#EF4444'
            ))
            
            fig.add_trace(go.Bar(
                name='Portfolio Weight',
                x=sector_analysis.index,
                y=sector_analysis['Total Weight %'],
                marker_color='#3B82F6'
            ))
            
            fig.update_layout(
                title="Sector Risk vs Weight Allocation",
                xaxis_title="Sector",
                yaxis_title="Percentage (%)",
                barmode='group',
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export full report
            if st.button("📊 Generate Full Report"):
                output = pd.ExcelWriter('risk_report.xlsx', engine='xlsxwriter')
                
                risk_metrics.to_excel(output, sheet_name='Risk Metrics', index=False)
                if portfolio_type == "Equal Weight":
                    rec_df.to_excel(output, sheet_name='Recommendations', index=False)
                sector_analysis.to_excel(output, sheet_name='Sector Analysis')
                
                output.close()
                
                with open('risk_report.xlsx', 'rb') as f:
                    st.download_button(
                        label="📥 Download Full Report (Excel)",
                        data=f,
                        file_name=f"bist50_risk_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()
