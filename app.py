import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta
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
    </style>
""", unsafe_allow_html=True)

class BIST50RiskAnalyzer:
    """Risk Budgeting Analysis for BIST 50 Stocks"""
    
    def __init__(self):
        self.tickers = [
            'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'EKGYO.IS',
            'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'HALKB.IS', 'ISCTR.IS',
            'KCHOL.IS', 'KOZAL.IS', 'KRDMD.IS', 'PETKM.IS', 'PGSUS.IS',
            'SAHOL.IS', 'SASA.IS', 'TCELL.IS', 'THYAO.IS', 'TOASO.IS'
        ]
        
        self.asset_names = {
            'AKBNK.IS': 'Akbank',
            'ARCLK.IS': 'Arçelik',
            'ASELS.IS': 'Aselsan',
            'BIMAS.IS': 'BİM',
            'EKGYO.IS': 'Emlak Konut',
            'EREGL.IS': 'Ereğli Demir Çelik',
            'FROTO.IS': 'Ford Otosan',
            'GARAN.IS': 'Garanti BBVA',
            'HALKB.IS': 'Halkbank',
            'ISCTR.IS': 'İş Bankası',
            'KCHOL.IS': 'Koç Holding',
            'KOZAL.IS': 'Koza Altın',
            'KRDMD.IS': 'Kardemir',
            'PETKM.IS': 'Petkim',
            'PGSUS.IS': 'Pegasus',
            'SAHOL.IS': 'Sabancı Holding',
            'SASA.IS': 'SASA Polyester',
            'TCELL.IS': 'Turkcell',
            'THYAO.IS': 'Türk Hava Yolları',
            'TOASO.IS': 'Tofaş'
        }
        
        self.sectors = {
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
        
        self.data_loaded = False
        
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_data(_self, start_date, end_date):
        """Fetch stock data with caching"""
        try:
            with st.spinner('📥 Veri indiriliyor...'):
                data = yf.download(
                    _self.tickers,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
                
                if data.empty:
                    st.error("Veri indirilemedi. Lütfen tarih aralığını kontrol edin.")
                    return None, None
                
                # Get adjusted close prices
                if 'Adj Close' in data:
                    prices = data['Adj Close']
                else:
                    prices = data['Close']
                
                # Calculate returns
                returns = prices.pct_change().dropna()
                
                return prices, returns
                
        except Exception as e:
            st.error(f"Veri indirme hatası: {str(e)}")
            return None, None
    
    def calculate_risk_metrics(self, returns, weights):
        """Calculate comprehensive risk metrics"""
        
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
            'Sembol': returns.columns,
            'Şirket': [self.asset_names.get(t, t) for t in returns.columns],
            'Sektör': [self.sectors.get(t, 'Diğer') for t in returns.columns],
            'Weight': weights,
            'Individual_Volatility': indiv_vol,
            'Marginal_Risk_Contribution': marginal_risk,
            'Component_Risk': component_risk,
            'Risk_Contribution_Pct': pct_contributions
        })
        
        # Sort by risk contribution
        risk_metrics = risk_metrics.sort_values('Risk_Contribution_Pct', ascending=False)
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
        risk_metrics['Diversification_Ratio'] = weighted_avg_vol / portfolio_volatility
        
        return risk_metrics, portfolio_volatility, cov_matrix
    
    def risk_parity_optimization(self, cov_matrix):
        """Calculate risk parity weights"""
        
        n = len(self.tickers)
        
        def risk_parity_objective(weights):
            weights = weights / weights.sum()
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
            return result.x / result.x.sum()
        else:
            return init_weights
    
    def calculate_risk_decomposition(self, returns, weights, risk_metrics):
        """Calculate systematic vs idiosyncratic risk decomposition"""
        
        portfolio_returns = returns @ weights
        
        # Calculate systematic and idiosyncratic components
        systematic_pct = []
        idiosyncratic_pct = []
        
        for i, ticker in enumerate(returns.columns):
            beta = risk_metrics[risk_metrics['Sembol'] == ticker]['Beta'].values[0]
            systematic_var = (beta ** 2) * portfolio_returns.var() * 252
            total_var = returns[ticker].var() * 252
            idiosyncratic_var = total_var - systematic_var
            
            systematic_pct.append((systematic_var / total_var) * 100)
            idiosyncratic_pct.append((idiosyncratic_var / total_var) * 100)
        
        risk_metrics['Systematic_Risk_%'] = systematic_pct
        risk_metrics['Idiosyncratic_Risk_%'] = idiosyncratic_pct
        
        return risk_metrics

def main():
    # Header
    st.markdown('<p class="main-header">📊 BIST 50 Risk Budgeting Dashboard</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Bu dashboard, BIST 50'de işlem gören 20 büyük hisse senedinden oluşan eşit ağırlıklı 
    portföyün **Marjinal Risk Katkıları (MRC)** ve **Risk Bütçeleme** analizini sunmaktadır.
    """)
    
    # Initialize analyzer
    analyzer = BIST50RiskAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Borsa_Istanbul_logo.svg/2560px-Borsa_Istanbul_logo.svg.png", 
                 width=200)
        
        st.markdown("## ⚙️ Parametreler")
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Başlangıç Tarihi",
                datetime(2020, 1, 1)
            )
        with col2:
            end_date = st.date_input(
                "Bitiş Tarihi",
                datetime.now()
            )
        
        # Portfolio type
        portfolio_type = st.radio(
            "Portföy Tipi",
            ["Eşit Ağırlıklı", "Risk Parity (Optimize)"],
            help="Eşit ağırlıklı veya risk parity optimizasyonu seçin"
        )
        
        # Risk metrics options
        st.markdown("### 📈 Gösterilecek Metrikler")
        show_individual_vol = st.checkbox("Bireysel Volatilite", value=True)
        show_mrc = st.checkbox("Marjinal Risk Katkısı (MRC)", value=True)
        show_beta = st.checkbox("Beta Katsayıları", value=True)
        show_systematic = st.checkbox("Sistematik Risk Dağılımı", value=True)
        
        # Download options
        st.markdown("### 📥 Rapor İndir")
        report_format = st.selectbox(
            "Dosya Formatı",
            ["Excel (.xlsx)", "CSV (.csv)"]
        )
        
        if st.button("📊 Rapor Oluştur", type="primary"):
            st.session_state['generate_report'] = True
        
        st.markdown("---")
        st.markdown("**Son Güncelleme:** " + datetime.now().strftime("%d.%m.%Y %H:%M"))
    
    # Main content area
    try:
        # Fetch data
        prices, returns = analyzer.fetch_data(start_date, end_date)
        
        if prices is not None and returns is not None:
            
            # Calculate portfolio weights
            if portfolio_type == "Eşit Ağırlıklı":
                weights = np.ones(len(analyzer.tickers)) / len(analyzer.tickers)
                st.session_state['portfolio_type'] = 'equal'
            else:
                # Calculate risk parity weights
                cov_matrix = returns.cov() * 252
                weights = analyzer.risk_parity_optimization(cov_matrix)
                st.session_state['portfolio_type'] = 'risk_parity'
            
            # Calculate risk metrics
            risk_metrics, portfolio_vol, cov_matrix = analyzer.calculate_risk_metrics(returns, weights)
            
            # Calculate risk decomposition
            if show_systematic:
                risk_metrics = analyzer.calculate_risk_decomposition(returns, weights, risk_metrics)
            
            # Display key metrics in cards
            st.markdown('<p class="sub-header">📌 Temel Portföy Metrikleri</p>', 
                       unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Portföy Volatilitesi (Yıllık)",
                    f"{portfolio_vol:.2%}",
                    delta=None
                )
            
            with col2:
                avg_vol = risk_metrics['Individual_Volatility'].mean()
                st.metric(
                    "Ortalama Bireysel Volatilite",
                    f"{avg_vol:.2%}",
                    delta=f"{(avg_vol - portfolio_vol):.2%}",
                    delta_color="inverse"
                )
            
            with col3:
                max_risk_contrib = risk_metrics.iloc[0]['Risk_Contribution_Pct']
                st.metric(
                    "En Yüksek Risk Katkısı",
                    f"{max_risk_contrib:.1f}%",
                    delta=risk_metrics.iloc[0]['Şirket']
                )
            
            with col4:
                div_ratio = risk_metrics['Diversification_Ratio'].iloc[0]
                st.metric(
                    "Çeşitlendirme Oranı",
                    f"{div_ratio:.2f}",
                    delta="İyi" if div_ratio > 1.5 else "Düşük",
                    delta_color="normal" if div_ratio > 1.5 else "inverse"
                )
            
            # Risk contribution visualization
            st.markdown('<p class="sub-header">🎯 Risk Katkı Dağılımı</p>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    y=risk_metrics['Şirket'],
                    x=risk_metrics['Risk_Contribution_Pct'],
                    orientation='h',
                    marker=dict(
                        color=risk_metrics['Risk_Contribution_Pct'],
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Risk %")
                    ),
                    text=risk_metrics['Risk_Contribution_Pct'].round(1).astype(str) + '%',
                    textposition='outside',
                    name='Risk Katkısı'
                ))
                
                # Add reference line for equal contribution
                equal_contrib = 100 / len(risk_metrics)
                fig.add_vline(x=equal_contrib, line_dash="dash", 
                             line_color="red", opacity=0.5,
                             annotation_text=f"Eşit Risk Hedefi ({equal_contrib:.1f}%)")
                
                fig.update_layout(
                    title="Hisse Bazında Risk Katkıları (Sıralanmış)",
                    xaxis_title="Risk Katkısı (%)",
                    yaxis_title="",
                    height=600,
                    showlegend=False,
                    hovermode='y'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart of top 5 vs others
                top_5 = risk_metrics.head(5)['Risk_Contribution_Pct'].sum()
                rest = 100 - top_5
                
                fig = go.Figure(data=[go.Pie(
                    labels=['İlk 5 Hisse', 'Diğer 15 Hisse'],
                    values=[top_5, rest],
                    hole=.4,
                    marker_colors=['#DC2626', '#6B7280']
                )])
                
                fig.update_layout(
                    title="Risk Yoğunlaşması",
                    annotations=[dict(text=f'%{top_5:.1f}', x=0.5, y=0.5, 
                                     font_size=20, showarrow=False)]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics table
            st.markdown('<p class="sub-header">📋 Detaylı Risk Metrikleri Tablosu</p>', 
                       unsafe_allow_html=True)
            
            # Prepare display table
            display_cols = ['Risk_Rank', 'Şirket', 'Sektör', 'Weight', 
                           'Risk_Contribution_Pct', 'Marginal_Risk_Contribution']
            
            if show_individual_vol:
                display_cols.append('Individual_Volatility')
            if show_mrc:
                display_cols.append('Marginal_Risk_Contribution')
            if show_beta:
                display_cols.append('Beta')
            if show_systematic:
                display_cols.extend(['Systematic_Risk_%', 'Idiosyncratic_Risk_%'])
            
            display_df = risk_metrics[display_cols].copy()
            
            # Format columns
            display_df['Weight'] = display_df['Weight'].map('{:.1%}'.format)
            display_df['Risk_Contribution_Pct'] = display_df['Risk_Contribution_Pct'].map('{:.1f}%'.format)
            
            if show_individual_vol:
                display_df['Individual_Volatility'] = display_df['Individual_Volatility'].map('{:.1%}'.format)
            if show_mrc:
                display_df['Marginal_Risk_Contribution'] = display_df['Marginal_Risk_Contribution'].map('{:.3f}'.format)
            if show_beta:
                display_df['Beta'] = display_df['Beta'].map('{:.2f}'.format)
            if show_systematic:
                display_df['Systematic_Risk_%'] = display_df['Systematic_Risk_%'].map('{:.1f}%'.format)
                display_df['Idiosyncratic_Risk_%'] = display_df['Idiosyncratic_Risk_%'].map('{:.1f}%'.format)
            
            # Rename columns for display
            column_names = {
                'Risk_Rank': 'Sıra',
                'Şirket': 'Şirket',
                'Sektör': 'Sektör',
                'Weight': 'Ağırlık',
                'Risk_Contribution_Pct': 'Risk Katkısı',
                'Marginal_Risk_Contribution': 'MRC',
                'Individual_Volatility': 'Bireysel Vol',
                'Beta': 'Beta',
                'Systematic_Risk_%': 'Sistematik Risk',
                'Idiosyncratic_Risk_%': 'Idiosenkratik Risk'
            }
            
            display_df = display_df.rename(columns=column_names)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Risk Katkısı": st.column_config.ProgressColumn(
                        "Risk Katkısı",
                        help="Toplam portföy riskine katkı yüzdesi",
                        format="%s",
                        min_value=0,
                        max_value=100,
                    )
                }
            )
            
            # Risk parity recommendations
            if portfolio_type == "Eşit Ağırlıklı":
                st.markdown('<p class="sub-header">⚖️ Risk Parity Optimizasyon Önerileri</p>', 
                           unsafe_allow_html=True)
                
                # Calculate risk parity weights
                rp_weights = analyzer.risk_parity_optimization(cov_matrix)
                
                # Calculate adjustments
                recommendations = []
                for i, ticker in enumerate(analyzer.tickers):
                    current_w = weights[i]
                    rp_w = rp_weights[i]
                    adjustment = rp_w - current_w
                    
                    recommendations.append({
                        'Şirket': analyzer.asset_names[ticker],
                        'Sektör': analyzer.sectors[ticker],
                        'Mevcut Ağırlık': f"{current_w:.1%}",
                        'Risk Parity Ağırlık': f"{rp_w:.1%}",
                        'Değişim': f"{adjustment:+.1%}",
                        'Öneri': 'AZALT' if adjustment < -0.005 else 'ARTTIR' if adjustment > 0.005 else 'KORU'
                    })
                
                rec_df = pd.DataFrame(recommendations)
                
                # Color coding for recommendations
                def color_recommendation(val):
                    if val == 'AZALT':
                        return 'background-color: #FEE2E2; color: #DC2626'
                    elif val == 'ARTTIR':
                        return 'background-color: #DCFCE7; color: #059669'
                    else:
                        return 'background-color: #F3F4F6; color: #6B7280'
                
                styled_rec = rec_df.style.applymap(color_recommendation, subset=['Öneri'])
                
                st.dataframe(
                    styled_rec,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                if st.button("📥 Optimizasyon Sonuçlarını İndir"):
                    csv = rec_df.to_csv(index=False)
                    st.download_button(
                        label="Excel formatında indir",
                        data=csv,
                        file_name=f"risk_parity_onerileri_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            # Risk decomposition visualization
            if show_systematic:
                st.markdown('<p class="sub-header">🔄 Sistematik vs Idiosenkratik Risk Dağılımı</p>', 
                           unsafe_allow_html=True)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Sistematik Risk',
                    x=risk_metrics['Şirket'],
                    y=risk_metrics['Systematic_Risk_%'],
                    marker_color='#2563EB'
                ))
                
                fig.add_trace(go.Bar(
                    name='Idiosenkratik Risk',
                    x=risk_metrics['Şirket'],
                    y=risk_metrics['Idiosyncratic_Risk_%'],
                    marker_color='#9CA3AF'
                ))
                
                fig.update_layout(
                    title="Risk Kompozisyonu Analizi",
                    xaxis_title="",
                    yaxis_title="Risk Yüzdesi (%)",
                    barmode='stack',
                    height=500,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Sector analysis
            st.markdown('<p class="sub-header">🏭 Sektörel Risk Analizi</p>', 
                       unsafe_allow_html=True)
            
            sector_risk = risk_metrics.groupby('Sektör').agg({
                'Risk_Contribution_Pct': 'sum',
                'Weight': 'sum'
            }).round(1)
            
            sector_risk.columns = ['Toplam Risk Katkısı %', 'Toplam Ağırlık %']
            sector_risk = sector_risk.sort_values('Toplam Risk Katkısı %', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    sector_risk,
                    values='Toplam Risk Katkısı %',
                    names=sector_risk.index,
                    title="Sektörel Risk Dağılımı",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=sector_risk.index,
                    y=sector_risk['Toplam Risk Katkısı %'],
                    name='Risk Katkısı',
                    marker_color='#EF4444'
                ))
                
                fig.add_trace(go.Bar(
                    x=sector_risk.index,
                    y=sector_risk['Toplam Ağırlık %'],
                    name='Portföy Ağırlığı',
                    marker_color='#3B82F6'
                ))
                
                fig.update_layout(
                    title="Sektör Bazında Risk vs Ağırlık",
                    xaxis_title="",
                    yaxis_title="Yüzde (%)",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Export functionality
            if 'generate_report' in st.session_state and st.session_state['generate_report']:
                st.markdown("---")
                st.markdown("### 📄 Rapor Önizleme")
                
                # Create Excel file in memory
                output = pd.ExcelWriter('risk_raporu.xlsx', engine='xlsxwriter')
                
                # Write sheets
                risk_metrics.to_excel(output, sheet_name='Risk Metrikleri', index=False)
                if portfolio_type == "Eşit Ağırlıklı":
                    rec_df.to_excel(output, sheet_name='Optimizasyon Önerileri', index=False)
                sector_risk.to_excel(output, sheet_name='Sektörel Analiz')
                
                # Save
                output.close()
                
                with open('risk_raporu.xlsx', 'rb') as f:
                    st.download_button(
                        label="📥 Tam Raporu İndir (Excel)",
                        data=f,
                        file_name=f"bist50_risk_raporu_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                st.session_state['generate_report'] = False
    
    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
        st.info("Lütfen daha sonra tekrar deneyin veya parametreleri değiştirin.")

if __name__ == "__main__":
    main()
