import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Portfolio Optimization Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS kustom untuk UI yang lebih baik
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #2c3e50;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.4rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 15px;
        border: none;
        padding: 0.4rem 1.5rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
    .tabs-container {
        margin-top: 1rem;
    }
    .stSpinner > div > div > svg {
        width: 2em;
        height: 2em;
    }
</style>
""", unsafe_allow_html=True)

# Judul
st.markdown('<h1 class="main-header">📈 Portfolio Optimization Tool</h1>', unsafe_allow_html=True)

# Fungsi Bantuan untuk Mengambil Data
@st.cache_data
def get_stock_data(tickers, start_date=None, end_date=None, period=None):
    """Ambil data saham dari Yahoo Finance"""
    try:
        if len(tickers) == 1:
            # Ambil data untuk satu ticker
            if period:
                data = yf.download(tickers[0], period=period)
            else:
                data = yf.download(tickers[0], start=start_date, end=end_date)
            
            if data.empty:
                return None
                
            df = pd.DataFrame({tickers[0]: data['Close']})
            return df.dropna()
        else:
            # Ambil data untuk beberapa ticker
            if period:
                data = yf.download(tickers, period=period)
            else:
                data = yf.download(tickers, start=start_date, end=end_date)
            
            if data.empty:
                return None
                
            # Ambil hanya kolom 'Close'
            if len(tickers) > 1:
                close_df = data['Close'].copy()
                close_df.columns = tickers  # Set kolom sesuai dengan ticker
            else:
                close_df = pd.DataFrame({tickers[0]: data['Close']})
            return close_df.dropna()
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

@st.cache_data
def get_index_data(index_ticker, start_date=None, end_date=None, period=None):
    """Ambil data indeks untuk perbandingan"""
    try:
        if period:
            data = yf.download(index_ticker, period=period)
        else:
            data = yf.download(index_ticker, start=start_date, end=end_date)
        
        if data.empty:
            return None
            
        # Ambil hanya kolom 'Close' dan pastikan hasilnya adalah Series
        close_data = data['Close'].copy()
        return close_data.dropna()
    except Exception as e:
        st.error(f"Error fetching index data: {str(e)}")
        return None

def calculate_returns(prices):
    """Hitung return harian"""
    return prices.pct_change().dropna()

def calculate_portfolio_metrics(returns, weights):
    """Hitung metrik portofolio"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_std if portfolio_std != 0 else 0
    
    # Hitung downside deviation untuk semi-varian
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.sqrt(np.dot(weights.T, np.dot(downside_returns.cov() * 252, weights)))
        sortino_ratio = portfolio_return / downside_std if downside_std != 0 else 0
    else:
        downside_std = 0
        sortino_ratio = 0
    
    return {
        'Annual Return': portfolio_return,
        'Annual Volatility': portfolio_std,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Downside Deviation': downside_std
    }

def mean_variance_optimization(returns, target_return=None):
    """Optimisasi Mean-Variance"""
    n_assets = len(returns.columns)
    
    def objective(weights):
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if target_return:
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(returns.mean() * x) * 252 - target_return})
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else initial_guess

def mean_semivariance_optimization(returns, target_return=None):
    """Optimisasi Mean-Semivariance"""
    n_assets = len(returns.columns)
    downside_returns = returns[returns < 0].fillna(0)
    
    def objective(weights):
        downside_cov = downside_returns.cov() * 252
        return np.sqrt(np.dot(weights.T, np.dot(downside_cov, weights)))
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if target_return:
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(returns.mean() * x) * 252 - target_return})
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else initial_guess

def minimum_variance_optimization(returns):
    """Optimisasi Minimum Variance"""
    n_assets = len(returns.columns)
    
    def objective(weights):
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else initial_guess

def generate_efficient_frontier(returns, num_portfolios=100):
    """Hasilkan Efficient Frontier"""
    n_assets = len(returns.columns)
    results = np.zeros((3, num_portfolios))
    
    target_returns = np.linspace(returns.mean().min() * 252, returns.mean().max() * 252, num_portfolios)
    
    for i, target in enumerate(target_returns):
        weights = mean_variance_optimization(returns, target)
        results[0, i] = target
        results[1, i] = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        results[2, i] = results[0, i] / results[1, i] if results[1, i] != 0 else 0
    
    return results

def plot_efficient_frontier(returns, portfolio_metrics=None):
    """Plot Efficient Frontier"""
    frontier = generate_efficient_frontier(returns)
    
    fig = go.Figure()
    
    # Efficient frontier
    fig.add_trace(go.Scatter(
        x=frontier[1], y=frontier[0],
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='#1f77b4', width=3)
    ))
    
    # Titik portofolio saat ini
    if portfolio_metrics:
        fig.add_trace(go.Scatter(
            x=[portfolio_metrics['Annual Volatility']],
            y=[portfolio_metrics['Annual Return']],
            mode='markers',
            name='Portofolio Saat Ini',
            marker=dict(color='#ff7f0e', size=12, symbol='star')
        ))
    
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatilitas Tahunan',
        yaxis_title='Return Tahunan',
        template='plotly_white',
        height=450,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def calculate_portfolio_value(returns, weights):
    """Hitung nilai portofolio dari waktu ke waktu"""
    portfolio_returns = (returns * weights).sum(axis=1)
    portfolio_value = 100 * (1 + portfolio_returns).cumprod()
    return portfolio_value

# Sidebar
with st.sidebar:
    st.markdown('<h2 class="sub-header">🔧 Konfigurasi</h2>', unsafe_allow_html=True)
    
    # Pemilihan Aset
    st.markdown("### 📊 Aset Portfolio")
    asset_input = st.text_area(
        "Masukkan ticker aset (satu per baris):",
        value="AAPL\nMSFT\nGOOGL\nBTC-USD\nETH-USD",
        help="Contoh: AAPL (Apple), BTC-USD (Bitcoin)"
    )
    tickers = [ticker.strip().upper() for ticker in asset_input.split('\n') if ticker.strip()]
    
    # Pemilihan Indeks BenchmarK
    st.markdown("### 📈 Indeks Benchmark")
    index_options = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "FTSE 100": "^FTSE",
        "Nikkei 225": "^N225"
    }
    selected_index = st.selectbox("Pilih indeks benchmark:", list(index_options.keys()))
    index_ticker = index_options[selected_index]
    
    # Periode Data Historis
    st.markdown("### 📅 Periode Data")
    
    # Opsi untuk memilih periode preset atau tanggal kustom
    period_selection_type = st.radio(
        "Pilih jenis periode:",
        ("Periode Preset", "Tanggal Kustom"),
        key="period_type_radio"
    )

    start_date = None
    end_date = None
    period_preset = None

    if period_selection_type == "Periode Preset":
        period_preset = st.selectbox(
            "Pilih periode data historis:",
            ["1y", "2y", "3y", "5y", "max"],
            index=1,
            key="preset_period_select"
        )
    else:
        # Input tanggal kustom
        today = pd.to_datetime("today").date()
        default_start_date = today - pd.DateOffset(years=2)
        
        start_date = st.date_input("Tanggal Mulai:", value=default_start_date, key="start_date_input")
        end_date = st.date_input("Tanggal Akhir:", value=today, key="end_date_input")

        if start_date >= end_date:
            st.error("Tanggal Mulai harus sebelum Tanggal Akhir.")
            start_date = None
            end_date = None

    # Metode Optimasi
    opt_methods = ["Mean-Variance", "Mean-Semivariance", "Minimum Variance"]
    st.markdown("### ⚙️ Pengaturan Optimasi")
    selected_opt_method = st.selectbox("Metode Optimasi:", opt_methods)
    
    # Tombol Muat Data
    load_button = st.button("🔄 Muat Data", type="primary", help="Klik untuk memuat dan proses data")

# Session state untuk manajemen status
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'index_data' not in st.session_state:
    st.session_state.index_data = None
if 'optimal_weights' not in st.session_state:
    st.session_state.optimal_weights = None
if 'portfolio_metrics' not in st.session_state:
    st.session_state.portfolio_metrics = None

# Proses Muat Data
if load_button and tickers:
    if period_selection_type == "Tanggal Kustom" and (start_date is None or end_date is None):
        st.error("Harap masukkan tanggal mulai dan tanggal akhir yang valid.")
    else:
        with st.spinner("Mengambil data... Silakan tunggu"):
            if period_selection_type == "Periode Preset":
                st.session_state.stock_data = get_stock_data(tickers, period=period_preset)
                st.session_state.index_data = get_index_data(index_ticker, period=period_preset)
            else:  # Tanggal Kustom
                st.session_state.stock_data = get_stock_data(tickers, start_date=start_date, end_date=end_date)
                st.session_state.index_data = get_index_data(index_ticker, start_date=start_date, end_date=end_date)
            
        if st.session_state.stock_data is not None and not st.session_state.stock_data.empty:
            st.success(f"Data berhasil dimuat untuk {len(tickers)} aset & indeks {selected_index}")
        else:
            st.error("Gagal memuat data atau tidak ada data yang tersedia untuk periode yang dipilih. Periksa ticker atau rentang tanggal Anda.")

# Konten utama
if st.session_state.stock_data is not None and not st.session_state.stock_data.empty:
    stock_data = st.session_state.stock_data
    index_data = st.session_state.index_data # Ambil index_data dari session_state
    returns = calculate_returns(stock_data)
    
    # Tab untuk navigasi
    manual_tab, optimize_tab, performance_tab = st.tabs(
        ["📊 Portofolio Manual", "⚙️ Optimisasi Portofolio", "📈 Analisis Kinerja"]
    )
    
    with manual_tab:
        st.markdown('<h2 class="sub-header">Konfigurasi Portofolio Manual</h2>', unsafe_allow_html=True)
        
        # Input bobot portofolio
        weights_inp = {}
        total_weight = 0
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.caption("Atur proporsi portfolio (%)")
            for i, ticker in enumerate(stock_data.columns):
                # Inisialisasi bobot jika belum ada di session_state
                if f"manual_{ticker}" not in st.session_state:
                    st.session_state[f"manual_{ticker}"] = 100 / len(stock_data.columns)

                weight_val = st.slider(
                    f"{ticker} proportion",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state[f"manual_{ticker}"],  # Menggunakan nilai dari session_state
                    step=0.5,
                    key=f"manual_{ticker}"
                )
                weights_inp[ticker] = weight_val / 100
                total_weight += weight_val
                
            # Validasi total bobot
            weight_status = st.empty()
            if abs(total_weight - 100) > 0.1:
                weight_status.warning(f"Total bobot: {total_weight:.1f}% (harus 100%)")
            else:
                weight_status.success("Bobot portfolio valid!")
                
        # Kalkulasi metrik
        with col2:
            weights_array = np.array([weights_inp[ticker] for ticker in stock_data.columns])
            # Normalisasi bobot jika totalnya tidak 100% (karena slider bisa tidak pas 100%)
            if np.sum(weights_array) > 0: # Pastikan tidak ada pembagian dengan nol
                weights_array = weights_array / np.sum(weights_array)
            else: # Jika semua bobot 0, distribusikan secara merata
                weights_array = np.array([1/len(stock_data.columns)] * len(stock_data.columns))

            st.session_state.portfolio_metrics = calculate_portfolio_metrics(returns, weights_array)
            
            # Visualisasi pie chart
            fig_pie = px.pie(
                names=stock_data.columns,
                values=weights_array * 100,  # Tampilkan dalam persentase
                title="Alokasi Portofolio",
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Metrik portofolio
        if st.session_state.portfolio_metrics:
            st.markdown("### 📉 Metrik Kinerja Portofolio")
            cols = st.columns(len(st.session_state.portfolio_metrics))
            metrics = st.session_state.portfolio_metrics
            metric_labels = {
                'Annual Return': 'Return Tahunan',
                'Annual Volatility': 'Volatilitas Tahunan',
                'Sharpe Ratio': 'Sharpe Ratio',
                'Sortino Ratio': 'Sortino Ratio',
                'Downside Deviation': 'Downside Deviation'
            }
            
            for i, (key, value) in enumerate(metrics.items()):
                with cols[i]:
                    st.metric(
                        label=metric_labels[key],
                        value=f"{value:.2%}" if 'Return' in key or 'Volatility' in key else f"{value:.3f}",
                        help=f"Metrik: {metric_labels[key]}"
                    )
    
    with optimize_tab:
        st.markdown('<h2 class="sub-header">Optimisasi Portofolio</h2>', unsafe_allow_html=True)
        
        # Tombol untuk menjalankan optimisasi
        optimize_button = st.button("⚡ Jalankan Optimisasi", type="primary", key="optimize_button")
        
        if optimize_button:
            with st.spinner("Sedang mengoptimisasi portofolio..."):
                if selected_opt_method == "Mean-Variance":
                    st.session_state.optimal_weights = mean_variance_optimization(returns)
                elif selected_opt_method == "Mean-Semivariance":
                    st.session_state.optimal_weights = mean_semivariance_optimization(returns)
                else:  # Minimum Variance
                    st.session_state.optimal_weights = minimum_variance_optimization(returns)
        
        # Tampilkan hasil optimisasi
        if st.session_state.optimal_weights is not None:
            cols = st.columns([1, 1])
            
            with cols[0]:
                st.markdown(f"### ✅ Result Optimisasi {selected_opt_method}")
                st.markdown("#### Proporsi Optimal:")
                
                # Tampilkan bobot dalam bentuk persentase
                weights_dict = {ticker: weight for ticker, weight in zip(stock_data.columns, st.session_state.optimal_weights)}
                for ticker, weight in weights_dict.items():
                    st.info(f"**{ticker}**: {weight:.2%}")
                
                # Hitung metrik untuk portofolio optimal
                opt_metrics = calculate_portfolio_metrics(returns, st.session_state.optimal_weights)
                st.markdown("#### Metrik Kinerja:")
                st.write(
                    f"- **Return Tahunan**: {opt_metrics.get('Annual Return', 0):.2%}\n"
                    f"- **Volatilitas Tahunan**: {opt_metrics.get('Annual Volatility', 0):.2%}\n"
                    f"- **Sharpe Ratio**: {opt_metrics.get('Sharpe Ratio', 0):.3f}\n"
                    f"- **Sortino Ratio**: {opt_metrics.get('Sortino Ratio', 0):.3f}"
                )
                
                # Tombol untuk menerapkan hasil optimasi
                apply_button = st.button("💾 Terapkan ke Portofolio Manual", key="apply_button")
                if apply_button:
                    for ticker, weight in weights_dict.items():
                        st.session_state[f"manual_{ticker}"] = weight * 100
                    st.experimental_rerun()  # Rerun untuk memperbarui slider di tab manual
            
            with cols[1]:
                # Tampilkan penyebaran bobot pada pie chart
                opt_weights_arr = st.session_state.optimal_weights * 100
                fig_opt_pie = px.pie(
                    names=stock_data.columns,
                    values=opt_weights_arr,
                    title="Proporsi Portfolio Optimal",
                    color_discrete_sequence=px.colors.sequential.Viridis,
                    height=370
                )
                fig_opt_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_opt_pie, use_container_width=True)
                
                # Tampilkan efficient frontier
                frontier_fig = plot_efficient_frontier(returns, 
                                                      portfolio_metrics=opt_metrics if opt_metrics else None)
                st.plotly_chart(frontier_fig, use_container_width=True)
    
    with performance_tab:
        st.markdown('<h2 class="sub-header">Analisis Kinerja</h2>', unsafe_allow_html=True)
        
        if st.session_state.portfolio_metrics is not None:
            # Hitung bobot portofolio manual
            manual_weights = np.array([weights_inp[ticker] for ticker in stock_data.columns])
            # Normalisasi bobot manual
            if np.sum(manual_weights) > 0:
                manual_weights_normalized = manual_weights / np.sum(manual_weights)
            else:
                manual_weights_normalized = np.array([1/len(stock_data.columns)] * len(stock_data.columns))
            
            # Hitung nilai portofolio manual
            manual_portfolio_value = calculate_portfolio_value(returns, manual_weights_normalized)
            
            # Hitung nilai portofolio optimal jika ada
            optimal_portfolio_value = None
            if st.session_state.optimal_weights is not None:
                optimal_portfolio_value = calculate_portfolio_value(returns, st.session_state.optimal_weights)
            
            # Plot kinerja
            performance_fig = go.Figure()
            
            # Plot portofolio manual
            performance_fig.add_trace(go.Scatter(
                x=manual_portfolio_value.index,
                y=manual_portfolio_value,
                mode='lines',
                name='Portofolio Manual',
                line=dict(color='#1f77b4', width=2.5)
            ))
            
            # Plot portofolio optimal jika ada
            if optimal_portfolio_value is not None:
                performance_fig.add_trace(go.Scatter(
                    x=optimal_portfolio_value.index,
                    y=optimal_portfolio_value,
                    mode='lines',
                    name=f'Portofolio Optimal ({selected_opt_method})',
                    line=dict(color='#2ca02c', width=2.5)
                ))
            
            # Plot indeks benchmark jika ada dan valid
            if index_data is not None and not index_data.empty:
                # Temukan tanggal mulai yang sama untuk portofolio dan indeks
                # Ambil tanggal mulai dari portofolio manual (atau optimal jika manual tidak ada)
                # dan indeks, lalu pilih yang paling baru
                start_date_for_comparison = manual_portfolio_value.index.min()
                if optimal_portfolio_value is not None:
                    start_date_for_comparison = max(start_date_for_comparison, optimal_portfolio_value.index.min())
                start_date_for_comparison = max(start_date_for_comparison, index_data.index.min())

                # Filter data indeks dari tanggal mulai perbandingan
                normalized_index = index_data.loc[index_data.index >= start_date_for_comparison]
                
                # Normalisasi hanya jika nilai awal bukan nol dan ada data
                if not normalized_index.empty:
                    # FIX: Use .iloc[0] for robust access to the first element of a Series
                    first_index_value = normalized_index.iloc[0] 
                    if first_index_value.item() != 0:
                        index_norm = normalized_index / first_index_value * 100
                        performance_fig.add_trace(go.Scatter(
                            x=index_norm.index,
                            y=index_norm.values,
                            mode='lines',
                            name=f'{selected_index} Index',
                            line=dict(color='#d62728', width=2.5)
                        ))
                    else:
                        st.warning("Tidak dapat menampilkan indeks benchmark karena nilai awal adalah nol.")
                else:
                    st.warning("Tidak cukup data untuk menampilkan indeks benchmark setelah penyelarasan tanggal.")
            else:
                st.warning("Data indeks benchmark tidak tersedia atau kosong.")
            
            performance_fig.update_layout(
                title='Perbandingan Kinerja Portofolio',
                xaxis_title='Tanggal',
                yaxis_title='Nilai Portofolio (Base = 100)',
                template='plotly_white',
                height=500,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
            )
            
            st.plotly_chart(performance_fig, use_container_width=True)
        else:
            st.warning("Silakan kalkulasi portofolio manual terlebih dahulu untuk melihat analisis kinerja.")

# Penjelasan metrik apabila belum ada data
else:
    st.markdown("""
    <div style='background-color: #262730; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        <h3 style='color: #1f77b4; margin-top: 0; font-size: 1.4rem;'>Petunjuk Penggunaan</h3>
        <ol style='font-size: 1rem;'>
            <li><strong>Masukkan ticker aset</strong> di sidebar (contoh: AAPL untuk Apple, BTC-USD untuk Bitcoin)</li>
            <li><strong>Pilih indeks benchmark</strong> sebagai pembanding kinerja portofolio</li>
            <li><strong>Atur periode data historis</strong>: Anda bisa memilih periode preset atau memasukkan tanggal mulai dan akhir kustom.</li>
            <li><strong>Klik tombol 'Muat Data'</strong> untuk memulai proses</li>
            <li>Lanjutkan ke tab yang diinginkan untuk analisis lebih lanjut</li>
        </ol>
        <p style='font-size: 0.9rem; margin-bottom: 0;'><strong>Catatan:</strong> Untuk data crypto gunakan format BTC-USD (Bitcoin), ETH-USD (Ethereum), dll.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📊 Tentang Metrik Portofolio
    - **Return Tahunan**: Total pengembalian tahunan portofolio
    - **Volatilitas Tahunan**: Ukuran risiko/fluktuasi kinerja
    - **Sharpe Ratio**: Ukuran performa risiko-terkoreksi (semakin tinggi semakin baik)
    - **Sortino Ratio**: Mirip Sharpe Ratio, hanya memperhitungkan risiko penurunan
    - **Downside Deviation**: Ukuran volatilitas khusus untuk kerugian
    - **Efficient Frontier**: Representasi visual portofolio optimal untuk risiko-tertentu
    """)

# Informasi versi dan sumber data
st.divider()
st.caption("""
**Portfolio Optimization Tool** - v2.1 | Sumber data: Yahoo Finance 📊  
Metode Mean-Variance diperkenalkan oleh Harry Markowitz (1952) untuk optimasi portofolio. Algoritma menggunakan SciPy untuk optimisasi numerik.
""")
