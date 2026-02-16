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
import requests
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
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class BIST50DataFetcher:
    """Alternative data sources for BIST stocks"""
    
    @staticmethod
    def get_ticker_mappings():
        """Alternative ticker formats for BIST stocks"""
        return {
            # Different formats that might work
            'AKBNK.IS': ['AKBNK.IS', 'AKBNK.IS', 'AKBNK.IS'],
            'ARCLK.IS': ['ARCLK.IS', 'ARCLK.IS', 'ARCLK.IS'],
            'ASELS.IS': ['ASELS.IS', 'ASELS.IS', 'ASELS.IS'],
            'BIMAS.IS': ['BIMAS.IS', 'BIMAS.IS', 'BIMAS.IS'],
            'EKGYO.IS': ['EKGYO.IS', 'EKGYO.IS', 'EKGYO.IS'],
            'EREGL.IS': ['EREGL.IS', 'EREGL.IS', 'EREGL.IS'],
            'FROTO.IS': ['FROTO.IS', 'FROTO.IS', 'FROTO.IS'],
            'GARAN.IS': ['GARAN.IS', 'GARAN.IS', 'GARAN.IS'],
            'HALKB.IS': ['HALKB.IS', 'HALKB.IS', 'HALKB.IS'],
            'ISCTR.IS': ['ISCTR.IS', 'ISCTR.IS', 'ISCTR.IS'],
            'KCHOL.IS': ['KCHOL.IS', 'KCHOL.IS', 'KCHOL.IS'],
            'KOZAL.IS': ['KOZAL.IS', 'KOZAL.IS', 'KOZAL.IS'],
            'KRDMD.IS': ['KRDMD.IS', 'KRDMD.IS', 'KRDMD.IS'],
            'PETKM.IS': ['PETKM.IS', 'PETKM.IS', 'PETKM.IS'],
            'PGSUS.IS': ['PGSUS.IS', 'PGSUS.IS', 'PGSUS.IS'],
            'SAHOL.IS': ['SAHOL.IS', 'SAHOL.IS', 'SAHOL.IS'],
            'SASA.IS': ['SASA.IS', 'SASA.IS', 'SASA.IS'],
            'TCELL.IS': ['TCELL.IS', 'TCELL.IS', 'TCELL.IS'],
            'THYAO.IS': ['THYAO.IS', 'THYAO.IS', 'THYAO.IS'],
            'TOASO.IS': ['TOASO.IS', 'TOASO.IS', 'TOASO.IS']
        }
    
    @staticmethod
    @st.cache_data(ttl=86400)  # Cache for 24 hours
    def fetch_from_yfinance(ticker, start_date, end_date):
        """Fetch data from Yahoo Finance with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                if not hist.empty:
                    return hist['Close']
                time.sleep(1)  # Wait before retry
            except:
                time.sleep(2)
        return None
    
    @staticmethod
    def fetch_sample_data():
        """Provide sample/demo data when APIs fail"""
        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='B')
        np.random.seed(42)
        
        sample_prices = {}
        
        # Generate realistic price paths for demo
        for ticker in BIST50RiskAnalyzer.tickers:
            # Base price
            base_price = np.random.uniform(10, 100)
            # Generate random walk
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            sample_prices[ticker] = prices
        
        return pd.DataFrame(sample_prices, index=dates)

class BIST50RiskAnalyzer:
    """Risk Budgeting Analysis for BIST 50 Stocks"""
    
    # Static tickers list
    tickers = [
        'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'EKGYO.IS',
        'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'HALKB.IS', 'ISCTR.IS',
        'KCHOL.IS', 'KOZAL.IS', 'KRDMD.IS', 'PETKM.IS', 'PGSUS.IS',
        'SAHOL.IS', 'SASA.IS', 'TCELL.IS', 'THYAO.IS', 'TOASO.IS'
    ]
    
    asset_names = {
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
        self.data_fetcher = BIST50DataFetcher()
        self.data_loaded = False
        
    @st.cache_data(ttl=3600)
    def fetch_data(_self, start_date, end_date, use_demo_data=False):
        """Fetch stock data with multiple fallback options"""
        
        if use_demo_data:
            st.info("📊 Demo verisi kullanılıyor - Gerçek veri bağlantısı kurulamadı")
            prices = _self.data_fetcher.fetch_sample_data()
            returns = prices.pct_change().dropna()
            return prices, returns, "demo"
        
        try:
            with st.spinner('📥 Veri indiriliyor (Yahoo Finance)...'):
                # Try primary data source
                all_prices = {}
                failed_tickers = []
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, ticker in enumerate(_self.tickers):
                    status_text.text(f"İndiriliyor: {_self.asset_names.get(ticker, ticker)}")
                    
                    # Try different ticker formats
                    price_data = None
                    
                    # Try original ticker
                    price_data = _self.data_fetcher.fetch_from_yfinance(
                        ticker, start_date, end_date
                    )
                    
                    if price_data is not None:
                        all_prices[ticker] = price_data
                    else:
                        failed_tickers.append(ticker)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(_self.tickers))
                
                progress_bar.empty()
                status_text.empty()
                
                if len(all_prices) == 0:
                    # If all failed, try alternative approach
                    st.warning("Yahoo Finance bağlantısı başarısız, alternatif yöntem deneniyor...")
                    
                    # Try batch download
                    try:
                        data = yf.download(
                            _self.tickers,
                            start=start_date,
                            end=end_date,
                            progress=False,
                            auto_adjust=True,
                            timeout=10
                        )
                        
                        if not data.empty:
                            if 'Adj Close' in data:
                                prices = data['Adj Close']
                            elif 'Close' in data:
                                prices = data['Close']
                            else:
                                prices = data
                            
                            returns = prices.pct_change().dropna()
                            return prices, returns, "yfinance_batch"
                    except:
                        pass
                    
                    # If still failed, use demo data
                    st.warning("⚠️ Gerçek veri alınamadı. Demo verisi ile devam ediliyor.")
                    prices = _self.data_fetcher.fetch_sample_data()
                    returns = prices.pct_change().dropna()
                    return prices, returns, "demo"
                
                # Create DataFrame from successful downloads
                prices = pd.DataFrame(all_prices)
                
                if len(prices.columns) < len(_self.tickers):
                    st.warning(f"⚠️ {len(failed_tickers)} hisse indirilemedi. Mevcut verilerle devam ediliyor.")
                    if failed_tickers:
                        st.write("İndirilemeyen hisseler:", 
                                [f"{t} ({_self.asset_names.get(t, t)})" for t in failed_tickers])
                
                # Calculate returns
                returns = prices.pct_change().dropna()
                
                return prices, returns, "yfinance_individual"
                
        except Exception as e:
            st.error(f"Veri indirme hatası: {str(e)}")
            st.info("Demo verisi kullanılıyor...")
            prices = _self.data_fetcher.fetch_sample_data()
            returns = prices.pct_change().dropna()
            return prices, returns, "demo"
    
    def calculate_risk_metrics(self, returns, weights):
        """Calculate comprehensive risk metrics"""
        
        # Filter returns to only include columns that exist
        available_tickers = returns.columns.tolist()
        weights_dict = {t: w for t, w in zip(self.tickers, weights) if t in available_tickers}
        
        if not weights_dict:
            st.error("Hiçbir hisse için veri bulunamadı!")
            return None, None, None
        
        # Recalculate weights for available tickers
        available_weights = np.array(list(weights_dict.values()))
        available_weights = available_weights / available_weights.sum()
        available_tickers = list(weights_dict.keys())
        
        # Filter returns
        returns_filtered = returns[available_tickers]
        
        # Annualized covariance matrix
        cov_matrix = returns_filtered.cov() * 252
        
        # Portfolio metrics
        portfolio_variance = available_weights @ cov_matrix @ available_weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Individual volatilities
        indiv_vol = np.sqrt(np.diag(cov_matrix))
        
        # Marginal Risk Contributions (MRC)
        marginal_risk = (cov_matrix @ available_weights) / portfolio_volatility
        
        # Component Risk Contributions (CRC)
        component_risk = available_weights * marginal_risk
        
        # Percentage contributions
        pct_contributions = (component_risk / portfolio_volatility) * 100
        
        # Create DataFrame
        risk_metrics = pd.DataFrame({
            'Sembol': available_tickers,
            'Şirket': [self.asset_names.get(t, t) for t in available_tickers],
            'Sektör': [self.sectors.get(t, 'Diğer') for t in available_tickers],
            'Weight': available_weights,
            'Individual_Volatility': indiv_vol,
            'Marginal_Risk_Contribution': marginal_risk,
            'Component_Risk': component_risk,
            'Risk_Contribution_Pct': pct_contributions
        })
        
        # Sort by risk contribution
        risk_metrics = risk_metrics.sort_values('Risk_Contribution_Pct', ascending=False)
        risk_metrics['Risk_Rank'] = range(1, len(risk_metrics) + 1)
        
        # Calculate betas
        portfolio_returns = returns_filtered @ available_weights
        betas = []
        for col in available_tickers:
            cov = returns_filtered[col].cov(portfolio_returns) * 252
            beta = cov / portfolio_variance
            betas.append(beta)
        
        risk_metrics['Beta'] = betas
        
        # Calculate diversification ratio
        weighted_avg_vol = np.sum(available_weights * indiv_vol)
        risk_metrics['Diversification_Ratio'] = weighted_avg_vol / portfolio_volatility
        
        return risk_metrics, portfolio_volatility, cov_matrix
    
    def risk_parity_optimization(self, cov_matrix, tickers):
        """Calculate risk parity weights"""
        
        n = len(tickers)
        
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

def main():
    # Header
    st.markdown('<p class="main-header">📊 BIST 50 Risk Budgeting Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Info box about data source issues
    st.markdown("""
    <div class="info-box">
        <strong>🔔 Veri Kaynağı Bilgisi:</strong> Yahoo Finance API'sinde yaşanan güncel sorunlar nedeniyle 
        bazı hisse verilerine erişim sağlanamamaktadır. Uygulama otomatik olarak:
        <ul>
            <li>Önce gerçek veri kaynaklarını dener</li>
            <li>Başarısız olursa demo verisi ile çalışır</li>
            <li>Tüm hesaplamaları mevcut veriler üzerinden yapar</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = BIST50RiskAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🏢 BIST 50 Risk Analizi")
        
        st.markdown("### ⚙️ Parametreler")
        
        # Data source option
        use_demo = st.checkbox(
            "Demo Verisi Kullan (Hızlı Test)",
            value=False,
            help="Gerçek veri yerine örnek veri seti kullanır"
        )
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Başlangıç",
                datetime(2020, 1, 1)
            )
        with col2:
            end_date = st.date_input(
                "Bitiş",
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
        
        st.markdown("---")
        st.markdown("**Son Güncelleme:** " + datetime.now().strftime("%d.%m.%Y %H:%M"))
    
    # Main content area
    try:
        # Fetch data
        prices, returns, data_source = analyzer.fetch_data(start_date, end_date, use_demo)
        
        if prices is not None and returns is not None:
            
            # Show data source indicator
            if data_source == "demo":
                st.info("📊 **Demo Modu**: Örnek veri seti kullanılıyor. Gerçek veri için lütfen daha sonra tekrar deneyin.")
            else:
                st.success(f"✅ Veri başarıyla indirildi ({data_source})")
            
            # Calculate portfolio weights
            available_tickers = returns.columns.tolist()
            
            if portfolio_type == "Eşit Ağırlıklı":
                weights = np.ones(len(available_tickers)) / len(available_tickers)
                st.session_state['portfolio_type'] = 'equal'
            else:
                # Calculate risk parity weights
                cov_matrix = returns.cov() * 252
                weights = analyzer.risk_parity_optimization(cov_matrix, available_tickers)
                st.session_state['portfolio_type'] = 'risk_parity'
            
            # Calculate risk metrics
            risk_metrics, portfolio_vol, cov_matrix = analyzer.calculate_risk_metrics(returns, weights)
            
            if risk_metrics is None:
                st.error("Risk metrikleri hesaplanamadı. Lütfen farklı parametreler deneyin.")
                return
            
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
            
            # Show number of assets included
            st.caption(f"Analiz edilen hisse sayısı: {len(risk_metrics)} / 20")
            
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
                    labels=['İlk 5 Hisse', 'Diğer'],
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
            display_cols = ['Risk_Rank', 'Şirket', 'Sektör', 'Weight']
            
            if show_individual_vol:
                display_cols.append('Individual_Volatility')
            if show_mrc:
                display_cols.append('Marginal_Risk_Contribution')
            if show_beta:
                display_cols.append('Beta')
            
            display_df = risk_metrics[display_cols].copy()
            
            # Format columns
            display_df['Weight'] = display_df['Weight'].map('{:.1%}'.format)
            
            if show_individual_vol:
                display_df['Individual_Volatility'] = display_df['Individual_Volatility'].map('{:.1%}'.format)
            if show_mrc:
                display_df['Marginal_Risk_Contribution'] = display_df['Marginal_Risk_Contribution'].map('{:.3f}'.format)
            if show_beta:
                display_df['Beta'] = display_df['Beta'].map('{:.2f}'.format)
            
            # Rename columns for display
            column_names = {
                'Risk_Rank': 'Sıra',
                'Şirket': 'Şirket',
                'Sektör': 'Sektör',
                'Weight': 'Ağırlık',
                'Individual_Volatility': 'Bireysel Vol',
                'Marginal_Risk_Contribution': 'MRC',
                'Beta': 'Beta'
            }
            
            display_df = display_df.rename(columns=column_names)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Risk parity recommendations (only for equal weight)
            if portfolio_type == "Eşit Ağırlıklı" and not use_demo:
                st.markdown('<p class="sub-header">⚖️ Risk Parity Optimizasyon Önerileri</p>', 
                           unsafe_allow_html=True)
                
                # Calculate risk parity weights
                rp_weights = analyzer.risk_parity_optimization(cov_matrix, available_tickers)
                
                # Calculate adjustments
                recommendations = []
                for i, ticker in enumerate(available_tickers):
                    current_w = weights[i]
                    rp_w = rp_weights[i]
                    adjustment = rp_w - current_w
                    
                    recommendations.append({
                        'Şirket': analyzer.asset_names.get(ticker, ticker),
                        'Sektör': analyzer.sectors.get(ticker, 'Diğer'),
                        'Mevcut Ağırlık': f"{current_w:.1%}",
                        'Risk Parity Ağırlık': f"{rp_w:.1%}",
                        'Değişim': f"{adjustment:+.1%}",
                        'Öneri': 'AZALT' if adjustment < -0.005 else 'ARTTIR' if adjustment > 0.005 else 'KORU'
                    })
                
                rec_df = pd.DataFrame(recommendations)
                
                st.dataframe(
                    rec_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            # Download button
            if st.button("📥 Raporu İndir (CSV)"):
                csv = risk_metrics.to_csv(index=False)
                st.download_button(
                    label="CSV dosyasını indir",
                    data=csv,
                    file_name=f"bist50_risk_raporu_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
        st.exception(e)
        
        # Offer demo mode as fallback
        if st.button("📊 Demo Modunda Çalıştır"):
            st.rerun()

if __name__ == "__main__":
    main()
