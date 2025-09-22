import streamlit as st
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import date, timedelta
from scipy.stats import norm
from scipy.optimize import minimize
from joblib import Parallel, delayed
import warnings
import time
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

# --- CONFIGURAZIONE PAGINA STREAMLIT ---
st.set_page_config(
    page_title="Dashboard di Ottimizzazione del Portafoglio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STATO E FUNZIONI DI NAVIGAZIONE ---
if 'page' not in st.session_state:
    st.session_state.page = 'backtest'
if 'bl_views' not in st.session_state:
    st.session_state.bl_views = []
if 'fwd_bl_views' not in st.session_state:
    st.session_state.fwd_bl_views = []
if 'tickers_for_optimizer' not in st.session_state:
    st.session_state.tickers_for_optimizer = []

def go_to_optimizer():
    st.session_state.page = 'optimizer'

def go_to_backtest():
    st.session_state.page = 'backtest'

# --- STATO DELLA SESSIONE ---
if 'bl_views' not in st.session_state:
    st.session_state.bl_views = []

# --- FUNZIONI CORE (DAL NOTEBOOK) ---

warnings.filterwarnings("ignore")

# --- Funzioni per il modello ARIMA-GARCH ---
@st.cache_data(ttl=3600)
def find_best_arima_order(log_ret):
    """Trova l'ordine ottimale per un modello ARIMA usando un grid search basato su BIC."""
    best_bic = np.inf
    best_order = None
    
    p_range = range(1, 6)
    d_range = range(0, 2)
    q_range = range(1, 6)

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(log_ret, order=(p, d, q))
                    results = model.fit()
                    if results.bic < best_bic:
                        best_bic = results.bic
                        best_order = (p, d, q)
                except Exception:
                    continue
    
    if best_order is None:
        return (1, 1, 1)
        
    return best_order

def estimate_arima_garch_params(log_ret):
    """Stima i parametri ARIMA-GARCH e restituisce i residui standardizzati."""
    try:
        order = find_best_arima_order(log_ret)
        model_arima = ARIMA(log_ret, order=order)
        res_arima = model_arima.fit()
        
        resid_arima = res_arima.resid
        garch = arch_model(resid_arima, p=1, q=1, vol='Garch', dist='Normal')
        res_garch = garch.fit(disp='off')
        
        std_resid = res_garch.resid / res_garch.conditional_volatility
        return order, res_garch.params, std_resid.dropna()
    except Exception:
        return None, None, None

def run_arima_garch_simulation(ticker_data, n_sim, n_steps):
    """Simula i percorsi di prezzo usando un approccio ARIMA-GARCH, parallelizzato per ticker."""
    
    def simulate_path(ticker):
        data = ticker_data[ticker]
        log_ret, S0, order, empirical_dist = data['log_ret'], data['S0'], data['arima_order'], data['std_resid'].values

        try:
            arima_model = ARIMA(log_ret, order=order)
            arima_res = arima_model.fit()
            
            garch_model = arch_model(arima_res.resid, p=1, q=1, vol='Garch', dist='Normal')
            garch_res = garch_model.fit(disp='off')
            
            last_resid = arima_res.resid[-1]
            last_vol = garch_res.conditional_volatility[-1]

            omega, alpha, beta = garch_res.params['omega'], garch_res.params['alpha[1]'], garch_res.params['beta[1]']

            innovations = np.random.choice(empirical_dist, size=(n_steps, n_sim), replace=True)
            mean_forecast = arima_res.forecast(steps=n_steps)
            
            sim_returns = np.zeros((n_steps, n_sim))
            
            h, r = last_vol**2, last_resid
            for i in range(n_sim):
                h_sim, r_sim = h, r
                for t in range(n_steps):
                    h_sim = omega + alpha * r_sim**2 + beta * h_sim
                    sigma_t = np.sqrt(h_sim)
                    
                    mu_t = mean_forecast.iloc[t]
                    ret_t = mu_t + sigma_t * innovations[t, i]
                    sim_returns[t, i] = ret_t
                    r_sim = ret_t - mu_t
            
            price_paths = np.zeros((n_steps + 1, n_sim))
            price_paths[0, :] = S0
            price_paths[1:, :] = S0 * np.exp(np.cumsum(sim_returns, axis=0))
            
            return ticker, price_paths
        except Exception:
            return ticker, None

    results = Parallel(n_jobs=-1)(delayed(simulate_path)(ticker) for ticker in ticker_data.keys())
    return {ticker: path for ticker, path in results if path is not None}


def download_ticker_data_robust(tv, ticker, exchange, start_date, end_date, interval=Interval.in_daily, n_bars=5000, retries=50):
    """Downloads ticker data with multiple retry attempts using a shared TvDatafeed instance."""
    for attempt in range(retries):
        try:
            df = tv.get_hist(symbol=ticker, exchange=exchange, interval=interval, n_bars=n_bars)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index).date
                df.sort_index(inplace=True) # Assicura che l'indice sia ordinato
                df = df.loc[start_date:end_date]
                if not df.empty:
                    return ticker, df  # Successo
            # Se il df è nullo o vuoto, riprova dopo una breve attesa
            if attempt < retries - 1:
                time.sleep(attempt + 1)  # Attesa progressiva
        except Exception:
            if attempt < retries - 1:
                time.sleep(attempt + 1)  # Attesa progressiva
            # All'ultimo tentativo, non c'è bisogno di attendere, fallirà e basta
    return ticker, None # Ritorna None se tutti i tentativi falliscono

def neg_loglik_merton(params, r, dt):
    # I parametri in input (mu, sigma, lam) sono annualizzati.
    # Li de-annualizziamo per il calcolo della verosimiglianza sul periodo dt.
    mu_ann, sigma_ann, lam_ann, muJ, sigmaJ = params
    
    kappa = np.exp(muJ + 0.5 * sigmaJ**2) - 1
    
    # Calcola la densità di probabilità per la parte diffusiva e la parte di salto
    # I parametri vengono scalati per il periodo dt
    drift_dt = (mu_ann - 0.5 * sigma_ann**2 - lam_ann * kappa) * dt
    vol_dt = sigma_ann * np.sqrt(dt)
    
    p1 = (1 - lam_ann * dt) * norm.pdf(r, loc=drift_dt, scale=vol_dt)
    
    m_eff = drift_dt + muJ
    v_eff = vol_dt**2 + sigmaJ**2
    
    p2 = lam_ann * dt * norm.pdf(r, loc=m_eff, scale=np.sqrt(v_eff))
    
    # Somma le probabilità e calcola il log-likelihood negativo
    likelihood = p1 + p2
    # Aggiungiamo un floor per evitare log(0)
    return -np.sum(np.log(np.maximum(likelihood, 1e-20)))

def estimate_params_for_ticker(ticker, df_daily, annualization_factor):
    log_ret = np.log(df_daily['close'] / df_daily['close'].shift(1)).dropna()
    if log_ret.empty: return ticker, None
    
    dt = 1.0 / annualization_factor
    
    # Stime iniziali ANNUALIZZATE
    mu_hist_ann = log_ret.mean() * annualization_factor
    sigma_hist_ann = log_ret.std(ddof=1) * np.sqrt(annualization_factor)
    
    # Stime iniziali per i salti (lam, muJ, sigmaJ) non sono annualizzate
    # lam è la frequenza annuale attesa dei salti
    init = np.array([mu_hist_ann, sigma_hist_ann, 10, 0.0, 0.1])
    # Il bound di lambda (frequenza salti) deve essere legato al fattore di annualizzazione
    bounds = [(-np.inf, np.inf), (1e-6, None), (1e-6, annualization_factor), (-np.inf, np.inf), (1e-6, None)]
    
    # L'ottimizzatore ora trova direttamente i parametri ANNUALIZZATI
    res = minimize(neg_loglik_merton, init, args=(log_ret.values, dt), bounds=bounds, method='L-BFGS-B')
    
    # I parametri stimati sono già annualizzati, non è necessaria alcuna conversione
    estimated_params = res.x
    
    return ticker, estimated_params

def run_simulation_for_ticker(ticker, params, S0, n_sim, n_steps, dt):
    # I parametri in input (mu, sigma, lam) sono già annualizzati.
    mu, sigma, lam, muJ, sigmaJ = params
    price_paths = np.zeros((n_steps + 1, n_sim)); price_paths[0] = S0
    kappa = np.exp(muJ + 0.5 * sigmaJ**2) - 1
    
    # Il drift viene calcolato con parametri annuali, poi scalato per dt nel ciclo.
    drift = mu - lam * kappa
    
    # Pre-calcola i rendimenti per una maggiore efficienza
    Z = np.random.standard_normal((n_steps, n_sim))
    N = np.random.poisson(lam * dt, (n_steps, n_sim))
    J_jumps = np.random.normal(muJ, sigmaJ, (n_steps, n_sim))
    J = N * J_jumps

    log_returns = (drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + J
    
    # Calcola i percorsi di prezzo in modo vettoriale
    price_paths[1:] = S0 * np.exp(np.cumsum(log_returns, axis=0))

    return ticker, price_paths

def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate, returns_df=None):
    portfolio_return = np.sum(expected_returns * weights)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if portfolio_vol == 0: return np.inf
    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
    return -sharpe

def neg_sortino_ratio(weights, expected_returns, risk_free_rate, returns_df):
    """Calcola il Sortino Ratio negativo. Richiede la serie di rendimenti."""
    portfolio_return = np.sum(expected_returns * weights)
    
    # Calcola la serie di rendimenti del portafoglio
    portfolio_returns_series = returns_df.dot(weights)
    
    # Calcola la deviazione standard dei rendimenti negativi (downside deviation)
    downside_returns = portfolio_returns_series[portfolio_returns_series < 0]
    downside_deviation = downside_returns.std()
    
    if downside_deviation == 0: return np.inf
    
    # Il Sortino Ratio non è annualizzato qui, ma è coerente per l'ottimizzazione
    sortino = (portfolio_return - risk_free_rate) / downside_deviation
    return -sortino

def optimize_portfolio(metric, expected_returns, cov_matrix, bounds, risk_free_rate, returns_df=None):
    num_assets = len(expected_returns)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    initial_weights = np.ones(num_assets) / num_assets
    
    if metric == 'Sortino Ratio':
        if returns_df is None:
            raise ValueError("La serie di rendimenti (returns_df) è necessaria per l'ottimizzazione del Sortino Ratio.")
        args = (expected_returns, risk_free_rate, returns_df)
        res = minimize(neg_sortino_ratio, initial_weights, args=args,
                       method='SLSQP', bounds=bounds, constraints=constraints)
    else: # Default a Sharpe Ratio
        args = (expected_returns, cov_matrix, risk_free_rate)
        res = minimize(neg_sharpe_ratio, initial_weights, args=args,
                       method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

def align_weights(weights, original_tickers, new_tickers_index):
    """
    Aligns portfolio weights from the optimization phase to the tickers available
    in the backtesting phase. Fills missing tickers with a weight of 0.
    """
    s = pd.Series(weights, index=original_tickers)
    aligned_s = s.reindex(new_tickers_index).fillna(0)
    
    # Re-normalize weights to sum to 1 if any assets were dropped.
    # This ensures the portfolio is fully invested.
    if aligned_s.sum() > 0 and not np.isclose(aligned_s.sum(), 1.0):
        aligned_s = aligned_s / aligned_s.sum()
        
    return aligned_s

def download_backtest_data_sequentially(tv, tickers_to_download, eval_start_date, eval_end_date, tv_interval, n_bars):
    """
    Downloads backtest data sequentially to avoid deadlocks with TvDatafeed and joblib.
    Provides progress updates in the Streamlit UI.
    """
    st.write("Starting sequential download for backtest data...")
    progress_bar = st.progress(0)
    all_prices_eval_dict = {}
    # tv = TvDatafeed() # REMOVED, instance is passed in

    for i, (ticker, exchange) in enumerate(tickers_to_download):
        progress_text = f"Downloading backtest data for {ticker} ({i+1}/{len(tickers_to_download)})..."
        progress_bar.progress((i + 1) / len(tickers_to_download), text=progress_text)
        
        try:
            # Using the core logic of get_hist directly, requesting a calculated number of bars
            df = tv.get_hist(symbol=ticker, exchange=exchange, interval=tv_interval, n_bars=n_bars)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index).date
                df.sort_index(inplace=True)
                df_filtered = df.loc[eval_start_date:eval_end_date]
                if not df_filtered.empty:
                    all_prices_eval_dict[ticker] = df_filtered['close']
            else:
                st.warning(f"No data returned for {ticker} during backtest period.")
        except Exception as e:
            st.warning(f"Failed to download {ticker} for backtest: {e}")
            
    progress_bar.empty()
    st.write("Sequential download complete.")
    return all_prices_eval_dict

# --- FUNZIONI DI PERFORMANCE ---
def calculate_performance_metrics(returns, risk_free_rate, lambda_aversion, weights, cov_matrix, annualization_factor=252):
    if returns.empty:
        return {
            'Annual Return': 0.0, 'Annual Volatility': 0.0, 'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0, 'Max Drawdown': 0.0, 'VaR (95%)': 0.0,
            'Diversification Ratio': np.nan, 'Effective Bets (ENB)': np.nan
        }
    
    # Calcoli base
    annualized_return = returns.mean() * annualization_factor
    annualized_volatility = returns.std() * np.sqrt(annualization_factor)
    
    # Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0.0
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(annualization_factor)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std != 0 else 0.0
    
    # Max Drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # VaR (Value at Risk)
    var_95 = returns.quantile(0.05)
    
    # Diversification Ratio (DR) e Effective Number of Bets (ENB)
    dr_value, enb_value = np.nan, np.nan
    if weights is not None and cov_matrix is not None and len(weights) > 0 and cov_matrix.shape[0] > 0:
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        weighted_asset_vols = np.sum(weights * np.sqrt(np.diag(cov_matrix)))
        
        if weighted_asset_vols > 0:
            dr_value = weighted_asset_vols / portfolio_vol
        
        # Effective Number of Bets (ENB) basato sulla concentrazione dei pesi
        if np.sum(weights**2) > 0:
            enb_value = 1 / np.sum(weights**2)

    return {
        'Annual Return': annualized_return,
        'Annual Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'VaR (95%)': var_95,
        'Diversification Ratio': dr_value,
        'Effective Bets (ENB)': enb_value
    }

def cov_to_corr(cov_matrix):
    """Converte una matrice di covarianza in una matrice di correlazione."""
    corr_matrix = cov_matrix.copy()
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if i != j:
                corr_matrix[i, j] = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])
    return corr_matrix

# --- FUNZIONE BLACK-LITTERAM ---
def calculate_black_litterman_returns(prior_returns, future_cov_matrix, final_tickers_for_analysis, bl_views, sectors_map, annualization_factor):
    st.info("Starting Black-Litterman calculation...")

    # Controlla se ci sono view con confidenza > 0
    has_confident_views = any(v.get('confidence', 0) > 0 for v in bl_views)

    # Se non ci sono view o nessuna ha confidenza, restituisci i prior e la covarianza originale
    if not bl_views or not has_confident_views:
        st.warning("No confident views provided. BL returns will be based on the prior only. The covariance matrix remains the original one.")
        return prior_returns, future_cov_matrix

    with st.expander("Black-Litterman Prior Details"):
        st.write("Prior returns are the implied arithmetic returns from the Inverse Volatility portfolio, annualized.")
        st.dataframe(pd.Series(prior_returns, index=final_tickers_for_analysis, name="Annualized Implied Arithmetic Returns").to_frame().style.format('{:.2%}'))

    tau = 0.05 
    tau_sigma = tau * future_cov_matrix

    P, Q, omegas = [], [], []
    ticker_list = list(final_tickers_for_analysis)
    num_assets = len(ticker_list)
    valid_views_count = 0

    for i, view in enumerate(bl_views):
        p_row = np.zeros(num_assets)
        is_valid = False
        
        if view['view_type'] == 'Absolute':
            if view['target_type'] == 'Sector':
                tickers_in_sector = [t for t in ticker_list if sectors_map.get(t) == view['asset1']]
                if tickers_in_sector:
                    for t in tickers_in_sector: p_row[ticker_list.index(t)] = 1.0 / len(tickers_in_sector)
                    is_valid = True
            elif view['target_type'] == 'Ticker':
                if view['asset1'] in ticker_list:
                    p_row[ticker_list.index(view['asset1'])] = 1.0
                    is_valid = True

        elif view['view_type'] == 'Relative':
            if view['target_type'] == 'Ticker vs Ticker':
                if view['asset1'] in ticker_list and view['asset2'] in ticker_list:
                    p_row[ticker_list.index(view['asset1'])] = 1.0
                    p_row[ticker_list.index(view['asset2'])] = -1.0
                    is_valid = True
            elif view['target_type'] == 'Sector vs Sector':
                tickers1 = [t for t in ticker_list if sectors_map.get(t) == view['asset1']]
                tickers2 = [t for t in ticker_list if sectors_map.get(t) == view['asset2']]
                if tickers1 and tickers2:
                    for t in tickers1: p_row[ticker_list.index(t)] = 1.0 / len(tickers1)
                    for t in tickers2: p_row[ticker_list.index(t)] = -1.0 / len(tickers2)
                    is_valid = True
        
        if is_valid and view.get('confidence', 0) > 0:
            valid_views_count += 1
            # La view 'Q' è già annualizzata dall'input dell'utente
            Q.append(view['value'])
            P.append(p_row)
            
            # La confidenza viene usata per calcolare l'incertezza della view
            confidence = max(0.0001, min(view['confidence'] / 100.0, 0.9999))
            
            # L'incertezza della view (omega) deve essere proporzionale alla varianza di mercato della view stessa,
            # NON scalata per tau. Questo è l'errore chiave. Usando la covarianza completa, l'incertezza
            # della view è su una scala comparabile a quella del prior.
            view_variance = p_row.dot(future_cov_matrix).dot(p_row.T)
            
            # Se la confidenza è bassa, l'incertezza (omega_k) sarà grande, e la view peserà poco.
            # Se la confidenza è alta, l'incertezza sarà piccola, e la view peserà molto.
            omega_k = ((1.0 / confidence) - 1.0) * view_variance
            omegas.append(omega_k)

    if not P:
        st.warning("No valid views with confidence > 0 for the analyzed tickers. BL returns based on prior only.")
        return prior_returns, future_cov_matrix

    P, Q, Omega = np.array(P), np.array(Q), np.diag(omegas)
    
    with st.expander(f"Black-Litterman View Details ({valid_views_count} valid)"):
        st.write("P Matrix (Views):"); st.dataframe(pd.DataFrame(P, columns=ticker_list))
        st.write("Q Vector (View Returns):"); st.dataframe(pd.Series(Q, name="Q Value").to_frame())
        st.write("Omega Diagonal (View Uncertainty):"); st.dataframe(pd.Series(np.diag(Omega), name="Uncertainty ω").to_frame())

    tau_sigma_inv, omega_inv = np.linalg.inv(tau_sigma), np.linalg.inv(Omega)

    # Calcola la covarianza della stima dei rendimenti medi posteriori.
    # Questa NON è la covarianza totale da usare per l'ottimizzazione.
    posterior_mean_covariance = np.linalg.inv(tau_sigma_inv + P.T.dot(omega_inv).dot(P))

    # Calcola i rendimenti medi posteriori (la media della distribuzione predittiva).
    posterior_returns_bl = posterior_mean_covariance.dot(tau_sigma_inv.dot(prior_returns) + P.T.dot(omega_inv).dot(Q))

    # Calcola la covarianza predittiva totale. Questa è la somma della covarianza di mercato originale
    # e dell'incertezza nella nostra stima dei rendimenti medi.
    # QUESTA è la matrice corretta da usare per l'ottimizzazione del portafoglio.
    posterior_cov_matrix_bl = future_cov_matrix + posterior_mean_covariance
    
    with st.expander("Black-Litterman Posterior Results"):
        posterior_df = pd.Series(posterior_returns_bl, index=final_tickers_for_analysis, name="Annualized Posterior Returns").to_frame()
        st.write("Annualized Posterior Returns (the result of blending priors with your views):")
        st.dataframe(posterior_df.style.format('{:.2%}'))

    st.success("Black-Litterman calculation completed.")
    return posterior_returns_bl, posterior_cov_matrix_bl

# --- FUNZIONE PRINCIPALE DI ANALISI (Backtesting - Schermata 1) ---
def run_backtest_analysis(tickers_with_sectors, sim_start_date, sim_end_date, eval_start_date, eval_end_date, include_bl, bl_views, bl_constraint_mode, bl_custom_bounds, simulation_model, selected_benchmark, optimization_metric):
    # Valori hardcoded
    risk_free_rate = 0.02
    lambda_aversion = 2.0

    # --- Frequenza Hardcoded su Weekly ---
    data_frequency = "Weekly"
    annualization_factor = 52
    tv_interval = Interval.in_weekly
    
    initial_tickers = [(t, e) for t, e, s in tickers_with_sectors]
    sectors_map = {t: s for t, e, s in tickers_with_sectors}
    
    tv = TvDatafeed() # Instantiate ONCE and reuse

    # --- Calcolo n_bars dinamico per il periodo di simulazione ---
    days_diff_sim = (sim_end_date - sim_start_date).days
    if data_frequency == 'Daily':
        n_bars_sim = int(days_diff_sim * 1.5)
    elif data_frequency == 'Weekly':
        n_bars_sim = int(days_diff_sim / 7 * 1.2)
    else: # Monthly
        n_bars_sim = int(days_diff_sim / 30 * 1.2)
    n_bars_sim = max(n_bars_sim, 100)
    st.info(f"Requesting approx. {n_bars_sim} bars for simulation period based on frequency '{data_frequency}'.")

    st.write(f"Starting sequential download of {data_frequency} data for {len(initial_tickers)} tickers (simulation period)...")
    progress_bar_sim = st.progress(0)
    hist_data = {}

    for i, (ticker, exchange) in enumerate(initial_tickers):
        progress_text = f"Downloading simulation data for {ticker} ({i+1}/{len(initial_tickers)})..."
        progress_bar_sim.progress((i + 1) / len(initial_tickers), text=progress_text)
        
        _, df = download_ticker_data_robust(tv, ticker, exchange, sim_start_date, sim_end_date, interval=tv_interval, n_bars=n_bars_sim)
        if df is not None and not df.empty:
            hist_data[ticker] = df
        else:
            st.warning(f"Could not download simulation data for {ticker}.")
            
    progress_bar_sim.empty()
    st.write("Simulation data download complete.")
    if not hist_data:
        st.error("Download failed for all tickers for the simulation period."); return None

    simulation_results = None
    n_steps = 1000 # Numero di passi di simulazione fisso per stabilità statistica
    st.info(f"Running simulation with {n_steps} steps.")

    if simulation_model == "Merton Jump-Diffusion":
        with st.spinner(f"Estimating Merton model parameters for {len(hist_data)} tickers..."):
            estimation_results = Parallel(n_jobs=-1)(delayed(estimate_params_for_ticker)(ticker, df, annualization_factor) for ticker, df in hist_data.items())
            mle_params = {ticker: params for ticker, params in estimation_results if params is not None}
            if not mle_params:
                st.error("Parameter estimation failed for all tickers."); return None

        with st.spinner(f"Running Merton simulations for {len(mle_params)} tickers..."):
            n_sim, dt = 100, 1 / annualization_factor
            simulation_args = [(ticker, params, hist_data[ticker]['close'].iloc[-1], n_sim, n_steps, dt) for ticker, params in mle_params.items()]
            simulation_results_list = Parallel(n_jobs=-1)(delayed(run_simulation_for_ticker)(*args) for args in simulation_args)
            simulation_results = {ticker: paths for ticker, paths in simulation_results_list if paths is not None}

    elif simulation_model == "ARIMA-GARCH":
        ticker_model_data = {}
        with st.spinner(f"Estimating ARIMA-GARCH parameters for {len(hist_data)} tickers..."):
            log_returns_map = {ticker: np.log(df['close'] / df['close'].shift(1)).dropna() for ticker, df in hist_data.items()}
            
            estimation_results = Parallel(n_jobs=-1)(delayed(estimate_arima_garch_params)(log_ret) for log_ret in log_returns_map.values())
            
            for (ticker, log_ret), (order, garch_params, std_resid) in zip(log_returns_map.items(), estimation_results):
                if order is not None and std_resid is not None and not std_resid.empty:
                    ticker_model_data[ticker] = {
                        'log_ret': log_ret,
                        'S0': hist_data[ticker]['close'].iloc[-1],
                        'arima_order': order,
                        'garch_params': garch_params,
                        'std_resid': std_resid
                    }
        
        if not ticker_model_data:
            st.error("ARIMA-GARCH parameter estimation failed for all tickers."); return None

        with st.spinner(f"Running ARIMA-GARCH simulations for {len(ticker_model_data)} tickers..."):
            n_sim = 100
            simulation_results = run_arima_garch_simulation(ticker_model_data, n_sim, n_steps)

    if not simulation_results:
        st.error("Monte Carlo simulation failed for all tickers."); return None

    final_tickers_for_analysis = sorted(list(simulation_results.keys()))
    num_assets = len(final_tickers_for_analysis)

    initial_ticker_names = {t[0] for t in initial_tickers}
    failed_tickers = initial_ticker_names - set(final_tickers_for_analysis)
    if failed_tickers:
        st.warning(f"⚠️ Analysis failed for: {', '.join(sorted(list(failed_tickers)))}")
    st.info(f"Successfully processed {num_assets} out of {len(initial_tickers)} initial tickers.")

    st.success(f"Successfully processed {num_assets} out of {len(initial_tickers)} initial tickers.")

    simulated_log_returns_df = pd.DataFrame({ticker: np.log(simulation_results[ticker][1:] / simulation_results[ticker][:-1]).flatten() for ticker in final_tickers_for_analysis})
    
    # La covarianza è calcolata correttamente sui log-returns
    future_cov_matrix = simulated_log_returns_df.cov() * annualization_factor
    
    # I rendimenti attesi devono essere aritmetici per coerenza con BL e Markowitz
    simulated_arithmetic_returns_df = np.exp(simulated_log_returns_df) - 1
    expected_returns_future = simulated_arithmetic_returns_df.mean() * annualization_factor

    with st.spinner(f"Optimizing portfolios for {num_assets} assets..."):
        # --- CALCOLO STRATEGIE ---

        # 1. Heuristic & Reference Portfolios (Non-Optimized in the traditional sense)
        future_vols = np.sqrt(np.diag(future_cov_matrix))
        weights_inverse_vol = (1 / future_vols) / np.sum(1 / future_vols)
        
        min_w_h, max_w_h = (1 / num_assets) / 2, (1 / num_assets) * 4
        
        # 2. Optimized Portfolios (calcolati ma non mostrati)
        weights_markowitz = optimize_portfolio(optimization_metric, expected_returns_future, future_cov_matrix, [(0, 1)] * num_assets, risk_free_rate, simulated_arithmetic_returns_df)
        weights_heuristic = optimize_portfolio(optimization_metric, expected_returns_future, future_cov_matrix, [(min_w_h, max_w_h)] * num_assets, risk_free_rate, simulated_arithmetic_returns_df)

        # 3. Equilibrium & Black-Litterman Portfolios
        # Calcola i rendimenti di equilibrio implicati dal portafoglio Inverse Volatility (il nostro "market cap")
        implied_log_returns = lambda_aversion * future_cov_matrix.dot(weights_inverse_vol)
        implied_arithmetic_prior = implied_log_returns + 0.5 * np.diag(future_cov_matrix)

        # Il portafoglio "Equilibrium" è il VERO prior del BL: un'ottimizzazione sui rendimenti implicati.
        weights_equilibrium = optimize_portfolio(optimization_metric, implied_arithmetic_prior, future_cov_matrix, [(0, 1)] * num_assets, risk_free_rate, simulated_arithmetic_returns_df)

        weights_map = {
            "Inverse Volatility": (weights_inverse_vol, final_tickers_for_analysis),
            "Equilibrium": (weights_equilibrium, final_tickers_for_analysis)
        }
        # Aggiungiamo Heuristic e Markowitz solo se necessario per debug o altre analisi, ma non verranno mostrati
        # weights_map["Heuristic"] = (weights_heuristic, final_tickers_for_analysis)
        # weights_map["Markowitz"] = (weights_markowitz, final_tickers_for_analysis)

        if include_bl:
            st.info("Calculating Black-Litterman portfolio...")
            
            has_confident_views = any(v.get('confidence', 0) > 0 for v in bl_views)

            if not has_confident_views:
                st.warning("No confident views for BL. Weights will be set to the Equilibrium portfolio.")
                weights_bl = weights_equilibrium
            else:
                # Il prior per BL sono i rendimenti aritmetici implicati.
                bl_expected_returns, bl_cov_matrix = calculate_black_litterman_returns(
                    implied_arithmetic_prior, 
                    future_cov_matrix, 
                    final_tickers_for_analysis, 
                    bl_views, 
                    sectors_map, 
                    annualization_factor
                )
                
                # Ottimizza il portafoglio BL con i rendimenti e la covarianza posterior
                if bl_constraint_mode == 'Custom':
                    bounds = [(bl_custom_bounds['min'], bl_custom_bounds['max'])] * num_assets
                elif bl_constraint_mode == 'Short-Selling Enabled':
                    bounds = [(-bl_custom_bounds['short_limit'], bl_custom_bounds['long_limit'])] * num_assets
                else: # Default is Heuristic
                    min_w_h, max_w_h = (1 / num_assets) / 2, (1 / num_assets) * 4
                    bounds = [(min_w_h, max_w_h)] * num_assets

                weights_bl = optimize_portfolio(optimization_metric, bl_expected_returns, bl_cov_matrix, bounds, risk_free_rate, simulated_arithmetic_returns_df)

            weights_map["Black-Litterman"] = (weights_bl, final_tickers_for_analysis)
        else:
            st.info("Numerical mode selected. Showing base strategies.")

    with st.spinner(f"Running backtest with {selected_benchmark[0]} as benchmark..."):
        original_ticker_tuples = {t: (t, e) for t, e, s in tickers_with_sectors}
        tickers_for_backtest_tuples = [original_ticker_tuples[t] for t in final_tickers_for_analysis] + [selected_benchmark]
        
        # --- Calcolo n_bars dinamico per il periodo di backtest ---
        days_diff_eval = (eval_end_date - eval_start_date).days
        if data_frequency == 'Daily':
            n_bars_eval = int(days_diff_eval * 1.5)
        elif data_frequency == 'Weekly':
            n_bars_eval = int(days_diff_eval / 7 * 1.2)
        else: # Monthly
            n_bars_eval = int(days_diff_eval / 30 * 1.2)
        n_bars_eval = max(n_bars_eval, 100)
        st.info(f"Requesting approx. {n_bars_eval} bars for backtest period (Weekly).")

        # Usa la nuova funzione di download sequenziale, passando l'istanza tv esistente e n_bars calcolato
        all_prices_eval_dict = download_backtest_data_sequentially(tv, tickers_for_backtest_tuples, eval_start_date, eval_end_date, tv_interval, n_bars_eval)
        
        if not all_prices_eval_dict:
            st.error("No data downloaded for the backtest period."); return None

        benchmark_prices = all_prices_eval_dict.pop(selected_benchmark[0], None)
        portfolio_returns_benchmark = benchmark_prices.pct_change().dropna() if benchmark_prices is not None else None
        
        prices_eval = pd.concat(all_prices_eval_dict, axis=1).reindex(columns=final_tickers_for_analysis).dropna(axis=1, how='all').dropna(axis=0, how='any')
        returns_eval = prices_eval.pct_change().dropna()
        if returns_eval.empty:
            st.error("The returns DataFrame for the backtest is empty."); return None

        valid_eval_tickers = returns_eval.columns
        aligned_weights = {k: align_weights(w, cols, valid_eval_tickers) for k, (w, cols) in weights_map.items()}
        
        returns_map = {k: returns_eval.dot(w) for k, w in aligned_weights.items()}
        
        # Calcola la matrice di covarianza del periodo di backtest per DR e ENB
        backtest_cov_matrix = returns_eval.cov() * annualization_factor
        
        performance_data = {}
        for name, returns in returns_map.items():
            weights = aligned_weights[name]
            # Allinea la matrice di covarianza ai ticker validi per il calcolo
            aligned_cov = backtest_cov_matrix.loc[weights.index, weights.index]
            performance_data[name] = calculate_performance_metrics(returns, risk_free_rate, lambda_aversion, weights.values, aligned_cov.values, annualization_factor)

        if portfolio_returns_benchmark is not None:
            # Per il benchmark, i pesi e la covarianza non sono applicabili nello stesso modo
            performance_data[f"Benchmark ({selected_benchmark[0]})"] = calculate_performance_metrics(portfolio_returns_benchmark, risk_free_rate, lambda_aversion, None, None, annualization_factor)
            
        performance_df = pd.DataFrame(performance_data).T

    summary_df = pd.DataFrame(index=final_tickers_for_analysis)
    for name, (weights, tickers) in weights_map.items():
        col_name = f'Weight {name.replace("-Weighted", "")}'
        summary_df[col_name] = pd.Series(weights, index=tickers)
    summary_df = summary_df.fillna(0).reset_index().rename(columns={'index': 'Ticker'})
    summary_df['Sector'] = summary_df['Ticker'].map(sectors_map)

    fig_perf = plt.figure(figsize=(15, 10))
    for name, returns in returns_map.items():
        (1 + returns).cumprod().plot(label=name, linewidth=2)
    if portfolio_returns_benchmark is not None:
        (1 + portfolio_returns_benchmark).cumprod().plot(label=f'Benchmark ({selected_benchmark[0]})', color='black', linewidth=2.5, linestyle='--')
    plt.title(f'Cumulative Portfolio Performance (Backtest: {eval_start_date} to {eval_end_date})', fontsize=16)
    plt.legend(); plt.grid(True)

    return summary_df, performance_df, fig_perf

# --- FUNZIONE DI OTTIMIZZAZIONE (Forward-Looking - Schermata 2) ---
def run_forward_looking_optimization(tickers_with_sectors, years_for_estimation, simulation_model, bl_views, bl_constraint_mode, bl_custom_bounds, optimization_metric):
    # --- 1. Impostazioni e Parametri ---
    risk_free_rate = 0.02
    lambda_aversion = 2.0
    
    # --- Frequenza Hardcoded su Weekly ---
    data_frequency = "Weekly"
    annualization_factor = 52
    tv_interval = Interval.in_weekly

    end_date = date.today()
    start_date = end_date - timedelta(days=int(years_for_estimation * 365.25))
    
    initial_tickers = [(t, e) for t, e, s in tickers_with_sectors]
    sectors_map = {t: s for t, e, s in tickers_with_sectors}

    tv = TvDatafeed() # Instantiate ONCE and reuse

    # --- 2. Download Dati ---
    # --- Calcolo n_bars per il periodo forward-looking (Weekly) ---
    days_diff_fwd = (end_date - start_date).days
    n_bars_fwd = int(days_diff_fwd / 7 * 1.2)
    n_bars_fwd = max(n_bars_fwd, 100)
    st.info(f"Requesting approx. {n_bars_fwd} bars for fwd-looking period (Weekly).")

    st.write(f"Starting sequential download of {data_frequency} data for {len(initial_tickers)} tickers ({years_for_estimation} years)...")
    progress_bar_fwd = st.progress(0)
    hist_data = {}

    for i, (ticker, exchange) in enumerate(initial_tickers):
        progress_text = f"Downloading fwd-looking data for {ticker} ({i+1}/{len(initial_tickers)})..."
        progress_bar_fwd.progress((i + 1) / len(initial_tickers), text=progress_text)
        
        _, df = download_ticker_data_robust(tv, ticker, exchange, start_date, end_date, interval=tv_interval, n_bars=n_bars_fwd)
        if df is not None and not df.empty:
            hist_data[ticker] = df
        else:
            st.warning(f"Could not download fwd-looking data for {ticker}.")
            
    progress_bar_fwd.empty()
    st.write("Forward-looking data download complete.")
    if not hist_data:
        st.error("Download failed for all tickers."); return None, None, None

    # --- 3. Stima Modello e Simulazione ---
    simulation_results = None
    n_steps = 1000 # Numero di passi di simulazione fisso per stabilità statistica
    st.info(f"Running simulation with {n_steps} steps.")
    
    if simulation_model == "Merton Jump-Diffusion":
        with st.spinner(f"Estimating Merton parameters..."):
            estimation_results = Parallel(n_jobs=-1)(delayed(estimate_params_for_ticker)(ticker, df, annualization_factor) for ticker, df in hist_data.items())
            mle_params = {ticker: params for ticker, params in estimation_results if params is not None}
        with st.spinner(f"Running Merton simulations..."):
            n_sim, dt = 100, 1/annualization_factor
            sim_args = [(t, p, hist_data[t]['close'].iloc[-1], n_sim, n_steps, dt) for t, p in mle_params.items()]
            sim_results_list = Parallel(n_jobs=-1)(delayed(run_simulation_for_ticker)(*args) for args in sim_args)
            simulation_results = {t: paths for t, paths in sim_results_list if paths is not None}

    elif simulation_model == "ARIMA-GARCH":
        ticker_model_data = {}
        with st.spinner(f"Estimating ARIMA-GARCH parameters..."):
            log_returns_map = {t: np.log(df['close'] / df['close'].shift(1)).dropna() for t, df in hist_data.items()}
            est_results = Parallel(n_jobs=-1)(delayed(estimate_arima_garch_params)(log_ret) for log_ret in log_returns_map.values())
            for (ticker, log_ret), (order, _, std_resid) in zip(log_returns_map.items(), est_results):
                if order and std_resid is not None and not std_resid.empty:
                    ticker_model_data[ticker] = {'log_ret': log_ret, 'S0': hist_data[ticker]['close'].iloc[-1], 'arima_order': order, 'std_resid': std_resid}
        with st.spinner(f"Running ARIMA-GARCH simulations..."):
            simulation_results = run_arima_garch_simulation(ticker_model_data, n_sim=100, n_steps=n_steps)

    if not simulation_results:
        st.error("Monte Carlo simulation failed."); return None, None, None

    final_tickers_for_analysis = sorted(list(simulation_results.keys()))
    num_assets = len(final_tickers_for_analysis)
    st.info(f"Successfully processed {num_assets} tickers for optimization.")

    sim_log_returns = pd.DataFrame({t: np.log(simulation_results[t][1:] / simulation_results[t][:-1]).flatten() for t in final_tickers_for_analysis})
    
    # La covarianza è calcolata correttamente sui log-returns
    future_cov_matrix = sim_log_returns.cov() * annualization_factor
    
    # I rendimenti attesi devono essere aritmetici per coerenza con BL e Markowitz
    sim_arithmetic_returns = np.exp(sim_log_returns) - 1
    expected_returns_future = sim_arithmetic_returns.mean() * annualization_factor
    
    # --- 4. Calcolo di TUTTE le strategie ---
    weights_map = {}
    bl_expected_returns, bl_cov_matrix = None, None # Inizializza
    
    with st.spinner("Optimizing portfolios..."):
        # Heuristic & Reference
        future_vols = np.sqrt(np.diag(future_cov_matrix))
        weights_map['Inverse Volatility'] = (1 / future_vols) / np.sum(1 / future_vols)
        
        min_w_h, max_w_h = (1 / num_assets) / 2, (1 / num_assets) * 4
        weights_map['Heuristic'] = optimize_portfolio(optimization_metric, expected_returns_future, future_cov_matrix, [(min_w_h, max_w_h)] * num_assets, risk_free_rate, sim_arithmetic_returns)

        # Markowitz (calcolato ma non mostrato)
        weights_markowitz = optimize_portfolio(optimization_metric, expected_returns_future, future_cov_matrix, [(0, 1)] * num_assets, risk_free_rate, sim_arithmetic_returns)
        
        # Equilibrium & Black-Litterman
        implied_log_returns_fwd = lambda_aversion * future_cov_matrix.dot(weights_map['Inverse Volatility'])
        implied_arithmetic_prior_fwd = implied_log_returns_fwd + 0.5 * np.diag(future_cov_matrix)
        weights_equilibrium = optimize_portfolio(optimization_metric, implied_arithmetic_prior_fwd, future_cov_matrix, [(0, 1)] * num_assets, risk_free_rate, sim_arithmetic_returns)
        weights_map['Equilibrium'] = weights_equilibrium
        # Non aggiungiamo Equilibrium a weights_map per non mostrarlo
        # weights_map['Equilibrium'] = weights_equilibrium


        has_confident_views_fwd = any(v.get('confidence', 0) > 0 for v in bl_views)

        if not has_confident_views_fwd:
            st.warning("No confident views for BL in forward-looking. Weights will be set to the Equilibrium portfolio.")
            weights_map['Black-Litterman'] = weights_map['Equilibrium']
        else:
            # Usa i rendimenti aritmetici implicati come prior per Black-Litterman
            bl_expected_returns, bl_cov_matrix = calculate_black_litterman_returns(
                implied_arithmetic_prior_fwd, 
                future_cov_matrix, 
                final_tickers_for_analysis, 
                bl_views, 
                sectors_map, 
                annualization_factor
            )
            
            if bl_expected_returns is None:
                st.error("Black-Litterman calculation failed."); return None, None, None
            
            # Ottimizza il portafoglio BL con i rendimenti e la covarianza posterior
            if bl_constraint_mode == 'Custom':
                bounds = [(bl_custom_bounds['min'], bl_custom_bounds['max'])] * num_assets
            elif bl_constraint_mode == 'Short-Selling Enabled':
                bounds = [(-bl_custom_bounds['short_limit'], bl_custom_bounds['long_limit'])] * num_assets
            else: # Default Heuristic per BL
                bounds = [(min_w_h, max_w_h)] * num_assets
            
            weights_map['Black-Litterman'] = optimize_portfolio(optimization_metric, bl_expected_returns, bl_cov_matrix, bounds, risk_free_rate, sim_arithmetic_returns)

    # --- 5. Preparazione Output ---
    # Tabella Pesi
    summary_weights_df = pd.DataFrame(index=final_tickers_for_analysis)
    for name, weights in weights_map.items():
        summary_weights_df[name] = weights
    summary_weights_df['Sector'] = summary_weights_df.index.map(sectors_map)

    # Tabella Performance
    performance_data = {}
    for name, weights in weights_map.items():
        # Usa i rendimenti e covarianza di BL solo per il portafoglio BL
        if name == "Black-Litterman" and bl_expected_returns is not None:
            ret, cov = bl_expected_returns, bl_cov_matrix
        else:
            ret, cov = expected_returns_future, future_cov_matrix
        
        portfolio_return = np.sum(ret * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        
        # Calcola il Sortino Ratio usando i rendimenti simulati
        portfolio_sim_returns = sim_arithmetic_returns.dot(weights)
        downside_returns = portfolio_sim_returns[portfolio_sim_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(annualization_factor)
        sortino_ratio = (portfolio_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        dr_value = np.sum(weights * np.sqrt(np.diag(cov))) / portfolio_vol if portfolio_vol > 0 else 0
        enb_value = 1 / np.sum(weights**2) if np.sum(weights**2) > 0 else 0

        performance_data[name] = {
            "Expected Annual Return": portfolio_return,
            "Expected Annual Volatility": portfolio_vol,
            "Expected Sharpe Ratio": (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0,
            "Expected Sortino Ratio": sortino_ratio,
            "Diversification Ratio": dr_value,
            "Effective Bets (ENB)": enb_value
        }
    
    performance_df = pd.DataFrame(performance_data).T
    
    # Grafici a Torta per ogni strategia
    pie_charts = {}
    for name, weights in weights_map.items():
        # Crea una serie di pesi con l'indice corretto per il groupby
        weight_series = pd.Series(weights, index=final_tickers_for_analysis)
        sector_alloc = weight_series.groupby(summary_weights_df['Sector']).sum()
        
        fig = px.pie(
            sector_alloc, values=sector_alloc.values, names=sector_alloc.index, 
            title=f'Sector Allocation: {name}', hole=.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        pie_charts[name] = fig

    return summary_weights_df, performance_df, pie_charts

# --- LAYOUT STREAMLIT ---

st.title("📊 Portfolio Optimization Dashboard")

if st.session_state.page == 'backtest':
    # --- SCHERMATA 1: BACKTESTING ---
    st.header("Screen 1: Comparative Backtesting")
    
    st.sidebar.header("⚙️ Backtest Settings")
    
    # --- Controlli Sidebar ---
    optimization_metric = st.sidebar.selectbox("Optimization Metric", ("Sharpe Ratio", "Sortino Ratio"), key="opt_metric_bt")
    simulation_model = st.sidebar.radio("Simulation Model", ("Merton Jump-Diffusion", "ARIMA-GARCH"), key="sim_model_bt")
    # Rimosso il selettore di frequenza, hardcoded su Weekly
    # data_frequency = st.sidebar.radio("Data Frequency", ("Daily", "Weekly", "Monthly"), key="freq_bt", help="Frequency for estimation and backtest. Note: Backtest is always on daily data for now.")
    
    st.sidebar.subheader("Select Tickers")
    sector_list = ["Unknown", "Technology (Hardware)", "Technology (Software & Services)", "Semiconductors", "Health (Pharmaceutical)", "Health (Biotech)", "Health (Medical Devices)", "Finance (Banks)", "Finance (Insurance)", "Finance (Asset Management)", "Industrial", "Aerospace & Defense", "Automotive", "Consumer Goods (Cyclical)", "Consumer Goods (Non-Cyclical)", "Retail", "Luxury", "Media & Entertainment", "Telecommunications", "Utilities", "Energy (Oil & Gas)", "Energy (Renewables)", "Commodities (Metals)", "Commodities (Agriculture)", "Real Estate", "Transport & Logistics"]
    default_tickers = [("NVDA", "NASDAQ", "Semiconductors"), ("AMD", "NASDAQ", "Semiconductors"), ("LLY", "NYSE", "Health (Pharmaceutical)"), ("CAL", "NYSE", "Consumer Goods (Non-Cyclical)"), ("WMT", "NYSE", "Retail"), ("MGY", "NYSE", "Energy (Oil & Gas)"), ("KR", "NYSE", "Retail"), ("FCN", "NYSE", "Technology (Software & Services)"), ("NYT", "NYSE", "Media & Entertainment"), ("TDS", "NYSE", "Telecommunications"), ("MUR", "NYSE", "Energy (Oil & Gas)"), ("STLA", "NYSE", "Automotive"), ("XOM", "NYSE", "Energy (Oil & Gas)"), ("ISP", "MIL", "Finance (Banks)"), ("ABEA", "XETR", "Technology (Software & Services)"), ("CCC3", "XETR", "Consumer Goods (Cyclical)"), ("RER1", "GETTEX", "Health (Pharmaceutical)"), ("DLN", "GETTEX", "Consumer Goods (Non-Cyclical)"), ("4BE", "GETTEX", "Finance (Banks)")]
    default_ticker_strings = [f"{t} ({e}) - {s}" for t, e, s in default_tickers]
    selected_ticker_strings = st.sidebar.multiselect("Choose tickers:", options=default_ticker_strings, default=default_ticker_strings, key=str(default_ticker_strings))

    st.sidebar.subheader("3. Add Tickers Manually")
    with st.sidebar.expander("Add up to 2 custom tickers"):
        col1, col2, col3 = st.columns([2,2,3]); manual_ticker_1 = col1.text_input("Ticker 1", placeholder="e.g. GOOGL"); manual_exchange_1 = col2.text_input("Exchange 1", placeholder="e.g. NASDAQ"); manual_sector_1 = col3.selectbox("Sector 1", sector_list, index=0)
        col4, col5, col6 = st.columns([2,2,3]); manual_ticker_2 = col4.text_input("Ticker 2", placeholder="e.g. ENEL"); manual_exchange_2 = col5.text_input("Exchange 2", placeholder="e.g. MIL"); manual_sector_2 = col6.selectbox("Sector 2", sector_list, index=0)

    st.sidebar.subheader("4. Analysis Parameters")
    years_for_estimation = st.sidebar.slider("Years for model estimation", 1, 10, 4, key="est_years_bt")
    years_for_backtest = st.sidebar.slider("Years for backtest", 1, 5, 1, key="bt_years_bt")
    
    benchmarks = {
        "S&P 500": ("SPY", "AMEX"),
        "FTSE 100": ("FTSE", "SPREADEX"),
        "EURO STOXX 50": ("EU50", "CAPITALCOM"),
        "DAX": ("DAX", "XETR")
    }
    benchmark_name = st.sidebar.selectbox("Select Benchmark", list(benchmarks.keys()))
    selected_benchmark = benchmarks[benchmark_name]

    st.sidebar.subheader("Strategies")
    include_bl = st.sidebar.checkbox("Include Black-Litterman Strategy", key="include_bl_bt")

    bl_constraint_mode = 'Default (Long-Only)'
    bl_custom_bounds = {'min': 0.0, 'max': 1.0, 'long_limit': 1.0, 'short_limit': 0.0}

    if include_bl:
        st.sidebar.subheader("5. Black-Litterman Constraints")
        bl_constraint_mode = st.sidebar.selectbox("Constraint Type", ['Heuristic (Default)', 'Custom', 'Short-Selling Enabled'], help="Define portfolio weight constraints.")

        if bl_constraint_mode == 'Short-Selling Enabled':
            short_limit = st.sidebar.slider("Max Short Weight (%)", 0.0, 100.0, 30.0, 1.0, help="Maximum weight for a short position.") / 100.0
            long_limit = 1.0 + short_limit
            st.sidebar.info(f"Long positions can go up to {long_limit:.0%}")
            bl_custom_bounds = {'short_limit': short_limit, 'long_limit': long_limit}
        elif bl_constraint_mode == 'Custom':
            min_w = st.sidebar.slider("Min Weight per Asset (%)", -100.0, 100.0, 0.0, 1.0) / 100.0
            max_w = st.sidebar.slider("Max Weight per Asset (%)", -100.0, 100.0, 25.0, 1.0) / 100.0
            bl_custom_bounds = {'min': min_w, 'max': max_w}

        st.sidebar.subheader("6. Black-Litterman Views")
        def add_view(): st.session_state.bl_views.append({"view_type": "Absolute", "target_type": "Sector", "asset1": None, "asset2": None, "value": 0.0, "confidence": 50.0, "id": time.time()})
        def remove_view(view_id): st.session_state.bl_views = [v for v in st.session_state.bl_views if v['id'] != view_id]; st.rerun()

        temp_final_tickers, temp_ticker_set = [], set()
        for ts in selected_ticker_strings:
            ticker, _, _ = ts.partition(" ("); temp_final_tickers.append(ticker); temp_ticker_set.add(ticker)
        if manual_ticker_1 and manual_ticker_1 not in temp_ticker_set: temp_final_tickers.append(manual_ticker_1.upper())
        if manual_ticker_2 and manual_ticker_2 not in temp_ticker_set: temp_final_tickers.append(manual_ticker_2.upper())
        all_available_sectors = sorted(list(set([s for _, _, s in default_tickers] + [manual_sector_1, manual_sector_2])))

        for i, view in enumerate(st.session_state.bl_views):
            st.sidebar.markdown(f"---"); st.sidebar.markdown(f"**View {i+1}**")
            col_type, col_del = st.sidebar.columns([4, 1])
            
            view['view_type'] = col_type.radio("View Type", ["Absolute", "Relative"], key=f"view_type_{view['id']}", horizontal=True)
            if col_del.button("🗑️", key=f"del_{view['id']}", use_container_width=True): remove_view(view['id'])

            if view['view_type'] == "Absolute":
                view['target_type'] = st.sidebar.radio("Target", ["Sector", "Ticker"], key=f"target_type_abs_{view['id']}", horizontal=True)
                view['asset1'] = st.sidebar.selectbox("Select", all_available_sectors if view['target_type'] == "Sector" else temp_final_tickers, key=f"asset1_abs_{view['id']}")
                view['asset2'] = None
            else: # Relative
                view['target_type'] = st.sidebar.radio("Target", ["Ticker vs Ticker", "Sector vs Sector"], key=f"target_type_rel_{view['id']}", horizontal=True)
                if view['target_type'] == "Ticker vs Ticker":
                    view['asset1'] = st.sidebar.selectbox("Outperforms", temp_final_tickers, key=f"asset1_rel_tic_{view['id']}")
                    view['asset2'] = st.sidebar.selectbox("Underperforms", temp_final_tickers, key=f"asset2_rel_tic_{view['id']}")
                else:
                    view['asset1'] = st.sidebar.selectbox("Outperforms", all_available_sectors, key=f"asset1_rel_sec_{view['id']}")
                    view['asset2'] = st.sidebar.selectbox("Underperforms", all_available_sectors, key=f"asset2_rel_sec_{view['id']}")

            st.session_state.bl_views[i]['value'] = st.sidebar.number_input("Expected Value (Q) (%)", value=view.get('value', 0.0) * 100, key=f"val_{view['id']}") / 100.0
            st.session_state.bl_views[i]['confidence'] = st.sidebar.slider("Confidence (%)", 0.0, 100.0, value=view.get('confidence', 50.0), key=f"conf_{view['id']}")

        st.sidebar.markdown("---")
        if st.sidebar.button("➕ Add View", use_container_width=True): add_view(); st.rerun()

    run_button = st.sidebar.button("🚀 Run Backtest Analysis", use_container_width=True)

    if run_button:
        final_tickers, ticker_set = [], set()
        for ts in selected_ticker_strings:
            ticker, rest = ts.split(" (", 1); exchange, sector = rest.rsplit(") - ", 1)
            if ticker not in ticker_set: final_tickers.append((ticker, exchange, sector)); ticker_set.add(ticker)
        if manual_ticker_1 and manual_exchange_1 and manual_ticker_1 not in ticker_set: final_tickers.append((manual_ticker_1.upper(), manual_exchange_1.upper(), manual_sector_1)); ticker_set.add(manual_ticker_1)
        if manual_ticker_2 and manual_exchange_2 and manual_ticker_2 not in ticker_set: final_tickers.append((manual_ticker_2.upper(), manual_exchange_2.upper(), manual_sector_2)); ticker_set.add(manual_ticker_2)

        if not final_tickers:
            st.warning("Please select at least one ticker for the analysis.")
        else:
            st.info(f"Analysis in progress for {len(final_tickers)} tickers: {', '.join([t[0] for t in final_tickers])}")
            today = date.today()
            eval_end_date = today; eval_start_date = today - timedelta(days=int(years_for_backtest * 365.25))
            sim_end_date = eval_start_date; sim_start_date = sim_end_date - timedelta(days=int(years_for_estimation * 365.25))
            
            # La frequenza è ora hardcoded su Weekly
            analysis_results = run_backtest_analysis(final_tickers, sim_start_date, sim_end_date, eval_start_date, eval_end_date, include_bl, st.session_state.bl_views, bl_constraint_mode, bl_custom_bounds, simulation_model, selected_benchmark, optimization_metric)

            if analysis_results:
                st.session_state.tickers_for_optimizer = final_tickers
                summary_df, performance_df, fig_perf = analysis_results
                st.success("Analysis completed successfully!")
                tab1, tab2, tab3 = st.tabs(["⚖️ Weights & Sector Allocation", "📈 Portfolio Performance", "📄 Metrics Table"])

                with tab1:
                    st.header("Optimized Weights and Sector Allocation")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Weights by Ticker")
                        style_cols = {col: '{:.2%}' for col in summary_df.columns if 'Weight' in col}
                        st.dataframe(summary_df.style.format(style_cols), height=35 * (len(summary_df) + 1))
                    with col2:
                        st.subheader("Allocation by Sector")
                        with st.expander("View Sector Allocation Pies", expanded=False):
                            for col in summary_df.columns:
                                if 'Weight' in col:
                                    sector_alloc = summary_df.groupby('Sector')[col].sum()
                                    if not sector_alloc.empty:
                                        fig_pie = px.pie(sector_alloc, values=sector_alloc.values, names=sector_alloc.index, title=f'Portfolio {col.replace("Weight ", "")}', hole=.3)
                                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                                        st.plotly_chart(fig_pie, use_container_width=True)

                with tab2:
                    st.header("Backtest: Cumulative Performance")
                    with st.expander("View Performance Chart", expanded=True):
                        st.pyplot(fig_perf)
                
                with tab3:
                    st.header("Performance Metrics Summary")
                    st.dataframe(performance_df.style.format({
                        "Annual Return": '{:.2%}', 
                        "Annual Volatility": '{:.2%}', 
                        "Max Drawdown": '{:.2%}', 
                        "VaR (95%)": '{:.2%}',
                        "Sharpe Ratio": '{:.2f}', 
                        "Sortino Ratio": '{:.2f}', 
                        "Diversification Ratio": '{:.2f}', 
                        "Effective Bets (ENB)": '{:.2f}'
                    }))

                st.write("---")
                st.button("➡️ Go to Forward-Looking Optimization", on_click=go_to_optimizer, use_container_width=True)

elif st.session_state.page == 'optimizer':
    # --- SCHERMATA 2: OTTIMIZZAZIONE FUTURE ---
    st.header("Screen 2: Forward-Looking Optimization")
    st.button("⬅️ Back to Backtesting", on_click=go_to_backtest)
    
    st.sidebar.header("⚙️ Optimizer Settings")
    
    if 'preliminary_analysis_done' not in st.session_state:
        st.session_state.preliminary_analysis_done = False

    if not st.session_state.tickers_for_optimizer:
        st.warning("Please run a backtest first to select the tickers.")
    else:
        tickers_info = ", ".join([t[0] for t in st.session_state.tickers_for_optimizer])
        st.info(f"Optimizing for: {tickers_info}")

        # --- FASE 1: ANALISI PRELIMINARE ---
        st.subheader("Phase 1: Preliminary Historical Analysis")
        st.markdown("Before setting your views, analyze the historical performance of the selected assets to get a sense of their returns. This will help you set more realistic expectations (Q values) for the Black-Litterman model.")
        
        years_for_hist_analysis = st.slider("Years for Historical Analysis", 1, 10, 3, key="hist_analysis_years")
        
        if st.button("Run Historical Analysis", key="run_hist_analysis"):
            with st.spinner("Downloading daily data for historical analysis..."):
                end_date_hist = date.today()
                start_date_hist = end_date_hist - timedelta(days=int(years_for_hist_analysis * 365.25))
                
                tv_hist = TvDatafeed() # Instantiate once for this analysis
                hist_data_analysis = {}
                for t, e, s in st.session_state.tickers_for_optimizer:
                    _, df = download_ticker_data_robust(tv_hist, t, e, start_date_hist, end_date_hist, interval=Interval.in_daily)
                    if df is not None and not df.empty:
                        hist_data_analysis[t] = df

                if hist_data_analysis:
                    log_returns_analysis = pd.DataFrame({t: np.log(df['close'] / df['close'].shift(1)) for t, df in hist_data_analysis.items()})
                    # Annualizziamo sempre per 252 per avere un'idea dei rendimenti su base annua
                    annualized_returns = log_returns_analysis.mean() * 252
                    st.session_state.historical_returns = annualized_returns
                    st.session_state.preliminary_analysis_done = True
                else:
                    st.error("Failed to download data for historical analysis.")
                    st.session_state.preliminary_analysis_done = False

        if st.session_state.preliminary_analysis_done and 'historical_returns' in st.session_state:
            st.success("Historical analysis complete. You can now proceed to Phase 2.")
            st.write("Historical Annualized Returns:")
            st.dataframe(st.session_state.historical_returns.to_frame('Annualized Return').style.format('{:.2%}'))
            st.markdown("---")

        # --- FASE 2: OTTIMIZZAZIONE ---
        if st.session_state.preliminary_analysis_done:
            st.subheader("Phase 2: Portfolio Optimization")

            optimization_metric_fwd = st.sidebar.selectbox("Optimization Metric", ("Sharpe Ratio", "Sortino Ratio"), key="opt_metric_fwd")
            simulation_model_fwd = st.sidebar.radio("Simulation Model", ("ARIMA-GARCH", "Merton Jump-Diffusion"), key="sim_model_fwd")
            years_for_estimation_fwd = st.sidebar.slider("Years of data for estimation", 1, 10, 5, key="est_years_fwd")
            
            st.sidebar.subheader("Black-Litterman Constraints")
            bl_constraint_mode_fwd = st.sidebar.selectbox("Constraint Type", ['Heuristic (Default)', 'Custom', 'Short-Selling Enabled'], key="bl_mode_fwd")
            
            bl_custom_bounds_fwd = {'min': 0.0, 'max': 1.0, 'long_limit': 1.0, 'short_limit': 0.0}
            if bl_constraint_mode_fwd == 'Short-Selling Enabled':
                short_limit_fwd = st.sidebar.slider("Max Short Weight (%)", 0.0, 100.0, 30.0, 1.0, key="short_fwd") / 100.0
                long_limit_fwd = 1.0 + short_limit_fwd
                st.sidebar.info(f"Long positions can go up to {long_limit_fwd:.0%}")
                bl_custom_bounds_fwd = {'short_limit': short_limit_fwd, 'long_limit': long_limit_fwd}
            elif bl_constraint_mode_fwd == 'Custom':
                min_w_fwd = st.sidebar.slider("Min Weight per Asset (%)", -100.0, 100.0, 0.0, 1.0, key="min_w_fwd") / 100.0
                max_w_fwd = st.sidebar.slider("Max Weight per Asset (%)", -100.0, 100.0, 25.0, 1.0, key="max_w_fwd") / 100.0
                bl_custom_bounds_fwd = {'min': min_w_fwd, 'max': max_w_fwd}

            st.sidebar.subheader("Black-Litterman Views")
            def add_fwd_view(): st.session_state.fwd_bl_views.append({"view_type": "Absolute", "target_type": "Sector", "asset1": None, "asset2": None, "value": 0.0, "confidence": 50.0, "id": time.time()})
            def remove_fwd_view(view_id): st.session_state.fwd_bl_views = [v for v in st.session_state.fwd_bl_views if v['id'] != view_id]; st.rerun()

            fwd_tickers_with_sectors = st.session_state.tickers_for_optimizer
            fwd_ticker_list = [t[0] for t in fwd_tickers_with_sectors]
            fwd_sector_list = sorted(list(set([s for t, e, s in fwd_tickers_with_sectors])))

            for i, view in enumerate(st.session_state.fwd_bl_views):
                st.sidebar.markdown(f"---"); st.sidebar.markdown(f"**View {i+1}**")
                col_type, col_del = st.sidebar.columns([4, 1])
                
                view['view_type'] = col_type.radio("View Type", ["Absolute", "Relative"], key=f"fwd_view_type_{view['id']}", horizontal=True)
                if col_del.button("🗑️", key=f"fwd_del_{view['id']}", use_container_width=True): remove_fwd_view(view['id'])

                if view['view_type'] == "Absolute":
                    view['target_type'] = st.sidebar.radio("Target", ["Sector", "Ticker"], key=f"fwd_target_type_abs_{view['id']}", horizontal=True)
                    view['asset1'] = st.sidebar.selectbox("Select", fwd_sector_list if view['target_type'] == "Sector" else fwd_ticker_list, key=f"fwd_asset1_abs_{view['id']}")
                    view['asset2'] = None
                else: # Relative
                    view['target_type'] = st.sidebar.radio("Target", ["Ticker vs Ticker", "Sector vs Sector"], key=f"fwd_target_type_rel_{view['id']}", horizontal=True)
                    if view['target_type'] == "Ticker vs Ticker":
                        view['asset1'] = st.sidebar.selectbox("Outperforms", fwd_ticker_list, key=f"fwd_asset1_rel_tic_{view['id']}")
                        view['asset2'] = st.sidebar.selectbox("Underperforms", fwd_ticker_list, key=f"fwd_asset2_rel_tic_{view['id']}")
                    else:
                        view['asset1'] = st.sidebar.selectbox("Outperforms", fwd_sector_list, key=f"fwd_asset1_rel_sec_{view['id']}")
                        view['asset2'] = st.sidebar.selectbox("Underperforms", fwd_sector_list, key=f"fwd_asset2_rel_sec_{view['id']}")

                st.session_state.fwd_bl_views[i]['value'] = st.sidebar.number_input("Expected Value (Q) (%)", value=view.get('value', 0.0) * 100, key=f"fwd_val_{view['id']}") / 100.0
                st.session_state.fwd_bl_views[i]['confidence'] = st.sidebar.slider("Confidence (%)", 0.0, 100.0, value=view.get('confidence', 50.0), key=f"fwd_conf_{view['id']}")

            st.sidebar.markdown("---")
            if st.sidebar.button("➕ Add View", use_container_width=True, key="add_fwd_view"): add_fwd_view(); st.rerun()
            
            run_fwd_button = st.sidebar.button("📈 Calculate Optimal Weights", use_container_width=True)

            if run_fwd_button:
                with st.spinner("Calculating forward-looking optimal weights..."):
                    fwd_results = run_forward_looking_optimization(
                        st.session_state.tickers_for_optimizer,
                        years_for_estimation_fwd,
                        simulation_model_fwd,
                        st.session_state.fwd_bl_views,
                        bl_constraint_mode_fwd,
                        bl_custom_bounds_fwd,
                        optimization_metric_fwd
                    )
                
                if fwd_results:
                    summary_weights_df, performance_df, pie_charts = fwd_results
                    st.success("Forward-looking optimization complete!")
                    
                    tab_w, tab_p, tab_s = st.tabs(["⚖️ Optimal Weights", "📈 Expected Performance", "📊 Sector Allocations"])
                    
                    with tab_w:
                        st.header("Optimized Weights Comparison")
                        style_cols = {col: '{:.2%}' for col in summary_weights_df.columns if col != 'Sector'}
                        st.dataframe(summary_weights_df.style.format(style_cols), height=35 * (len(summary_weights_df) + 1))
                    
                    with tab_p:
                        st.header("Expected Performance Metrics")
                        st.dataframe(performance_df.style.format({
                            "Expected Annual Return": '{:.2%}',
                            "Expected Annual Volatility": '{:.2%}',
                            "Expected Sharpe Ratio": '{:.2f}',
                            "Expected Sortino Ratio": '{:.2f}',
                            "Diversification Ratio": '{:.2f}',
                            "Effective Bets (ENB)": '{:.2f}'
                        }))
                    
                    with tab_s:
                        st.header("Sector Allocation by Strategy")
                        for strategy_name, fig in pie_charts.items():
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to run forward-looking optimization.")
