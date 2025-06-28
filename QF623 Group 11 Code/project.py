import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt
import sys

def optimize_portfolio(returns, min_risk=0.03, risk_free_rate=0.0, long_short=False):
    """
    Optimize portfolio weights to maximize Sharpe Ratio
    subject to minimum annualized risk constraint.
    Args:
        returns (pd.DataFrame): Daily log returns, columns are tickers.
        min_risk (float): Minimum annualized portfolio risk (std dev).
        risk_free_rate (float): Risk-free rate for Sharpe ratio.
        long_short (bool): If True, allow long-short (weights in [-1,1], abs(weights).sum()=1).
    Returns:
        dict: {'weights': np.array, 'sharpe': float, 'risk': float, 'ret': float}
    """
    mean_daily = returns.mean()
    cov_daily = returns.cov()
    n = len(mean_daily)
    ann_factor = 252  # trading days per year

    def portfolio_stats(weights):
        port_ret = np.dot(weights, mean_daily) * ann_factor
        port_vol = np.sqrt(np.dot(weights, np.dot(cov_daily, weights))) * np.sqrt(ann_factor)
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0
        return port_ret, port_vol, sharpe

    def neg_sharpe(weights):
        return -portfolio_stats(weights)[2]

    if long_short:
        # abs(weights).sum() == 1, weights in [-1, 1]
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1},
            {'type': 'ineq', 'fun': lambda w: portfolio_stats(w)[1] - min_risk}
        ]
        bounds = [(-1, 1)] * n
        x0 = np.ones(n) / n
    else:
        # weights.sum() == 1, weights in [0, 1]
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: portfolio_stats(w)[1] - min_risk}
        ]
        bounds = [(0, 1)] * n
        x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    weights = result.x
    port_ret, port_vol, sharpe = portfolio_stats(weights)
    return {
        'weights': weights,
        'sharpe': sharpe,
        'risk': port_vol,
        'ret': port_ret
    }

def rolling_portfolio_backtest(returns, window=252, min_risk=0.03, risk_free_rate=0.0, long_short=False):
    """
    Rolling window portfolio optimization and out-of-sample backtest.
    For each day T, use data up to T to optimize, then apply weights to T+1 returns.
    Args:
        returns (pd.DataFrame): Daily log returns.
        window (int): Rolling window size (default 252 trading days).
        min_risk (float): Minimum annualized risk.
        risk_free_rate (float): Risk-free rate.
        long_short (bool): If True, allow long-short.
    Returns:
        pd.DataFrame: Portfolio returns, weights, Sharpe, risk, etc. for each rebalance.
    """
    results = []
    dates = returns.index
    for i in range(window, len(returns) - 1):
        window_returns = returns.iloc[i - window:i]
        try:
            opt = optimize_portfolio(window_returns, min_risk, risk_free_rate, long_short)
            weights = opt['weights']
            # Out-of-sample return: apply weights to T+1 returns
            next_ret = returns.iloc[i + 1].values
            port_ret = np.dot(weights, next_ret)
            results.append({
                'date': dates[i + 1],
                'port_return': port_ret,
                'sharpe': opt['sharpe'],
                'risk': opt['risk'],
                'ret': opt['ret'],
                **{f'w_{col}': w for col, w in zip(returns.columns, weights)}
            })
        except Exception as e:
            # Optimization failed, skip this period
            results.append({
                'date': dates[i + 1],
                'port_return': np.nan,
                'sharpe': np.nan,
                'risk': np.nan,
                'ret': np.nan,
                **{f'w_{col}': np.nan for col in returns.columns}
            })
    df = pd.DataFrame(results)
    return df.set_index('date')

def download_factor_returns(start, end):
    """
    Download daily returns for common macro factors using yfinance.
    Factors: Market (SPY), Bonds (TLT), Gold (GLD), USD (UUP), Oil (USO)
    Returns:
        pd.DataFrame: columns = ['MKT', 'BOND', 'GOLD', 'USD', 'OIL']
    """
    tickers = {
        'MKT': 'SPY',    # US Equity Market
        'BOND': 'TLT',   # Long-term US Treasuries
        'GOLD': 'GLD',   # Gold
        'USD': 'UUP',    # US Dollar Index
        'OIL': 'USO'     # Oil
    }
    data = yf.download(list(tickers.values()), start=start, end=end, interval="1d", auto_adjust=False)['Adj Close']
    # If only one ticker, data is a Series
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.ffill()
    returns = data.pct_change().dropna()
    returns.columns = [k for k in tickers.keys() if tickers[k] in returns.columns or returns.shape[1] == 1]
    return returns

def performance_attribution(portfolio_returns, factor_returns):
    """
    Regress portfolio returns on factor returns to estimate betas.
    Args:
        portfolio_returns (pd.Series): Portfolio daily returns (index: date).
        factor_returns (pd.DataFrame): DataFrame of factor returns (columns: factors, index: date).
    Returns:
        Regression summary and betas.
    """
    # Align dates
    aligned = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
    if aligned.empty:
        print("No overlapping dates or all data is NaN after alignment. Cannot run regression.")
        print("portfolio_returns shape:", portfolio_returns.shape, "index:", portfolio_returns.index.min(), "-", portfolio_returns.index.max())
        print("factor_returns shape:", factor_returns.shape, "index:", factor_returns.index.min(), "-", factor_returns.index.max())
        return None
    y = aligned.iloc[:, 0]
    X = aligned.iloc[:, 1:]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model.params

def run_full_performance_attribution(portfolio_results_csv):
    """
    Loads portfolio backtest results, downloads macro factor returns,
    and runs performance attribution regression.
    """
    # Load portfolio returns
    results = pd.read_csv(portfolio_results_csv, index_col=0, parse_dates=True)
    port_returns = results['port_return']

    # Get date range for factor download
    start, end = results.index.min(), results.index.max()

    # Download macro factor returns (Market, Bonds, Gold, USD, Oil)
    factors = download_factor_returns(start, end)

    # Run regression and print summary
    betas = performance_attribution(port_returns, factors)
    print("\nEstimated Betas to Macro Factors:")
    print(betas)

def download_fama_french_factors(start, end):
    """
    Download Fama-French 3-factor (or 5-factor) daily data from Ken French's website.
    Returns:
        pd.DataFrame: columns = ['MKT_RF', 'SMB', 'HML', 'RF', ...]
    """
    import pandas_datareader.data as web
    ff = web.DataReader('F-F_Research_Data_Factors_Daily', 'famafrench', start, end)[0]
    ff = ff / 100.0  # Convert percent to decimal
    ff.index = pd.to_datetime(ff.index)
    return ff

def advanced_performance_attribution(portfolio_returns, start, end, use_ff5=False):
    """
    Run advanced attribution using Fama-French 3-factor or 5-factor model.
    Args:
        portfolio_returns (pd.Series): Portfolio daily returns (index: date).
        start, end: date range for factor download.
        use_ff5 (bool): If True, use 5-factor model.
    Returns:
        Regression summary and betas.
    """
    # Download Fama-French factors
    ff = download_fama_french_factors(start, end)
    if use_ff5:
        # Download 5-factor data
        import pandas_datareader.data as web
        ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_Daily', 'famafrench', start, end)[0]
        ff5 = ff5 / 100.0
        ff5.index = pd.to_datetime(ff5.index)
        ff = ff5

    # Align and compute excess returns
    aligned = pd.concat([portfolio_returns, ff], axis=1).dropna()
    y = aligned.iloc[:, 0] - aligned['RF']  # Excess portfolio return
    if use_ff5:
        X = aligned[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    else:
        X = aligned[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model.params

def download_sector_factors(start, end):
    """
    Download daily returns for S&P 500 sector ETFs using yfinance.
    Returns:
        pd.DataFrame: columns = sector tickers (e.g., XLK, XLF, etc.)
    """
    sector_tickers = [
        "XLC",  # Communication Services
        "XLY",  # Consumer Discretionary
        "XLP",  # Consumer Staples
        "XLE",  # Energy
        "XLF",  # Financials
        "XLV",  # Health Care
        "XLI",  # Industrials
        "XLB",  # Materials
        "XLRE", # Real Estate
        "XLK",  # Technology
        "XLU"   # Utilities
    ]
    data = yf.download(sector_tickers, start=start, end=end, interval="1d", auto_adjust=False)['Adj Close']
    data = data.ffill()
    returns = data.pct_change().dropna()
    return returns

def download_custom_macro_factors(start, end, custom_tickers):
    """
    Download daily returns for custom macro factors using yfinance.
    Args:
        custom_tickers (dict): e.g., {'CHINA': 'FXI', 'EUROPE': 'VGK'}
    Returns:
        pd.DataFrame: columns = custom macro factor names
    """
    data = yf.download(list(custom_tickers.values()), start=start, end=end, interval="1d", auto_adjust=False)['Adj Close']
    data = data.ffill()
    returns = data.pct_change().dropna()
    returns.columns = [k for k in custom_tickers.keys()]
    return returns

def hedging_analysis(portfolio_returns, market_returns):
    """
    Evaluate the effect of hedging by regressing portfolio returns on market returns,
    then compute the market-neutral (hedged) portfolio return series.
    Args:
        portfolio_returns (pd.Series): Portfolio daily returns.
        market_returns (pd.Series): Market daily returns (e.g., SPY).
    Returns:
        pd.Series: Hedged portfolio returns.
    """
    aligned = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
    if aligned.empty:
        print("No overlapping dates or all data is NaN after alignment. Cannot run regression.")
        print("portfolio_returns shape:", portfolio_returns.shape, "index:", portfolio_returns.index.min(), "-", portfolio_returns.index.max())
        print("market_returns shape:", market_returns.shape, "index:", market_returns.index.min(), "-", market_returns.index.max())
        return None
    y = aligned.iloc[:, 0]
    X = sm.add_constant(aligned.iloc[:, 1])
    model = sm.OLS(y, X).fit()
    beta = model.params[1]
    hedged_returns = y - beta * aligned.iloc[:, 1]
    print(f"Estimated market beta: {beta:.4f}")
    print(f"Mean return (unhedged): {y.mean():.6f}")
    print(f"Mean return (hedged): {hedged_returns.mean():.6f}")
    return hedged_returns

def full_rich_attribution_and_hedging(portfolio_results_csv):
    """
    Combines macro, sector, and custom macro factors for attribution,
    and evaluates the effect of market-neutral (hedged) strategies.
    """
    # Load portfolio returns
    results = pd.read_csv(portfolio_results_csv, index_col=0, parse_dates=True)
    port_returns = results['port_return']
    start, end = results.index.min(), results.index.max()

    # Download macro factors
    macro_factors = download_factor_returns(start, end)

    # Download sector factors
    sector_factors = download_sector_factors(start, end)

    # Download custom macro factors (example: China and Europe ETFs)
    custom_factors = download_custom_macro_factors(start, end, {'CHINA': 'FXI', 'EUROPE': 'VGK'})

    # Combine all factors
    all_factors = pd.concat([macro_factors, sector_factors, custom_factors], axis=1)

    # Run performance attribution with all factors
    print("\n===== Full Attribution: Macro + Sector + Custom Factors =====")
    betas = performance_attribution(port_returns, all_factors)
    print("\nEstimated Betas to All Factors:")
    print(betas)

    # Hedging analysis: market-neutral returns
    print("\n===== Hedging Analysis: Market-Neutral Portfolio =====")
    market = macro_factors['MKT']
    hedged_returns = hedging_analysis(port_returns, market)

    # Optional: Attribution on hedged returns
    print("\n===== Attribution on Market-Neutral (Hedged) Returns =====")
    betas_hedged = performance_attribution(hedged_returns, all_factors)
    print("\nEstimated Betas to All Factors (Hedged):")
    print(betas_hedged)

def compute_momentum_signal(price_df, lookback=126):
    """
    Compute momentum signal for each ETF: past N-day return.
    """
    return price_df.pct_change(periods=lookback)

def compute_volatility_signal(price_df, lookback=21):
    """
    Compute volatility signal: negative rolling std of daily returns (lower vol is better).
    """
    daily_ret = price_df.pct_change()
    return -daily_ret.rolling(lookback).std()

def compute_short_term_reversal_signal(price_df, lookback=5):
    """
    Compute short-term reversal: negative past week return (recent losers favored).
    """
    return -price_df.pct_change(periods=lookback)

def compute_value_signal(price_df, lookback=252):
    """
    Compute value signal: price relative to 1-year moving average (lower = more value).
    """
    ma = price_df.rolling(lookback).mean()
    return -(price_df / ma - 1)

def compute_alpha_signals(
    price_df, 
    momentum_lookback=126, 
    value_lookback=252, 
    momentum_weight=1.0, 
    value_weight=0.0, 
    regime_threshold=None
):
    momentum = compute_momentum_signal(price_df, lookback=momentum_lookback)
    value = compute_value_signal(price_df, lookback=value_lookback)
    def zscore(df):
        return (df - df.mean(axis=1, skipna=True).values[:, None]) / df.std(axis=1, skipna=True).values[:, None]
    momentum_z = zscore(momentum)
    value_z = zscore(value)
    if regime_threshold is not None:
        avg_mom = momentum_z.mean(axis=1)
        alpha = pd.DataFrame(index=momentum_z.index, columns=momentum_z.columns)
        for date in momentum_z.index:
            if avg_mom.loc[date] < regime_threshold:
                alpha.loc[date] = value_z.loc[date]
            else:
                alpha.loc[date] = momentum_z.loc[date]
        return alpha
    else:
        # Simple blend
        return momentum_weight * momentum_z + value_weight * value_z

def rolling_alpha_backtest(
    price_df, window=252, min_risk=0.03, risk_free_rate=0.0, long_short=False, top_n=2,
    target_vol=False, benchmark_returns=None, alpha_signals=None,
    momentum_lookback=126, value_lookback=252, momentum_weight=1.0, value_weight=0.0, regime_threshold=None,
    transaction_cost=0.0, rebalance_freq='D'
):
    """
    Rolling window backtest using alpha signals for ETF selection.
    At each rebalance, select top-N ETFs by alpha score, then optimize portfolio.
    Optionally applies volatility targeting to match benchmark volatility.
    """
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    window = min(window, len(log_returns))
    if alpha_signals is None:
        alpha_signals = compute_alpha_signals(
            price_df,
            momentum_lookback=momentum_lookback,
            value_lookback=value_lookback,
            momentum_weight=momentum_weight,
            value_weight=value_weight,
            regime_threshold=regime_threshold
        )
    results = []
    dates = log_returns.index
    prev_weights = np.zeros(len(price_df.columns))
    # Only rebalance on specified frequency
    rebalance_dates = pd.Series(dates).asfreq(rebalance_freq).dropna().values if rebalance_freq != 'D' else dates
    for i in range(window, len(log_returns) - 1):
        if dates[i + 1] not in rebalance_dates:
            continue
        window_returns = log_returns.iloc[i - window:i]
        alpha_today = alpha_signals.iloc[i - 1]
        if long_short:
            # Select top-N for long, bottom-N for short
            longs = alpha_today.dropna().sort_values(ascending=False).head(top_n).index.tolist()
            shorts = alpha_today.dropna().sort_values(ascending=True).head(top_n).index.tolist()
            selected = list(set(longs + shorts))
            if len(selected) < 2:
                results.append({
                    'date': dates[i + 1],
                    'port_return': np.nan,
                    'sharpe': np.nan,
                    'risk': np.nan,
                    'ret': np.nan,
                    **{f'w_{col}': np.nan for col in price_df.columns}
                })
                continue
            # Assign +1 to longs, -1 to shorts, then normalize so abs(weights).sum()=1
            init_weights = np.array([1 if etf in longs else -1 for etf in selected])
            init_weights = init_weights / np.sum(np.abs(init_weights))
            try:
                opt = optimize_portfolio(window_returns[selected], min_risk, risk_free_rate, long_short=True)
                weights = opt['weights']
                next_ret = log_returns.iloc[i + 1][selected].values
                port_ret = np.dot(weights, next_ret)
                # Transaction cost: sum of abs(weight change) * cost
                curr_weights = np.zeros(len(price_df.columns))
                for idx, col in enumerate(price_df.columns):
                    if col in selected:
                        curr_weights[idx] = weights[selected.index(col)]
                turnover = np.sum(np.abs(curr_weights - prev_weights))
                port_ret -= turnover * transaction_cost
                prev_weights = curr_weights
                # Volatility targeting
                if target_vol and benchmark_returns is not None:
                    port_hist = log_returns.iloc[i-window+1:i+1][selected].values @ weights
                    realized_vol = np.std(port_hist)
                    bench_hist = benchmark_returns.iloc[i-window+1:i+1]
                    bench_vol = np.std(bench_hist)
                    if realized_vol > 0 and bench_vol > 0:
                        scale = bench_vol / realized_vol
                        port_ret *= scale
                weight_dict = {f'w_{col}': (weights[selected.index(col)] if col in selected else 0.0) for col in price_df.columns}
                results.append({
                    'date': dates[i + 1],
                    'port_return': port_ret,
                    'sharpe': opt['sharpe'],
                    'risk': opt['risk'],
                    'ret': opt['ret'],
                    **weight_dict
                })
            except Exception as e:
                results.append({
                    'date': dates[i + 1],
                    'port_return': np.nan,
                    'sharpe': np.nan,
                    'risk': np.nan,
                    'ret': np.nan,
                    **{f'w_{col}': np.nan for col in price_df.columns}
                })
        else:
            # Long-only: select top-N
            top_etfs = alpha_today.dropna().sort_values(ascending=False).head(top_n).index.tolist()
            if len(top_etfs) < 1:
                results.append({
                    'date': dates[i + 1],
                    'port_return': np.nan,
                    'sharpe': np.nan,
                    'risk': np.nan,
                    'ret': np.nan,
                    **{f'w_{col}': np.nan for col in price_df.columns}
                })
                continue
            try:
                if len(top_etfs) == 1:
                    weights = np.array([1.0])
                else:
                    opt = optimize_portfolio(window_returns[top_etfs], min_risk, risk_free_rate, long_short=False)
                    weights = opt['weights']
                next_ret = log_returns.iloc[i + 1][top_etfs].values
                port_ret = np.dot(weights, next_ret)
                # Transaction cost: sum of abs(weight change) * cost
                curr_weights = np.zeros(len(price_df.columns))
                for idx, col in enumerate(price_df.columns):
                    if col in top_etfs:
                        curr_weights[idx] = weights[top_etfs.index(col)]
                turnover = np.sum(np.abs(curr_weights - prev_weights))
                port_ret -= turnover * transaction_cost
                prev_weights = curr_weights
                # Volatility targeting
                if target_vol and benchmark_returns is not None:
                    port_hist = log_returns.iloc[i-window+1:i+1][top_etfs].values @ weights
                    realized_vol = np.std(port_hist)
                    bench_hist = benchmark_returns.iloc[i-window+1:i+1]
                    bench_vol = np.std(bench_hist)
                    if realized_vol > 0 and bench_vol > 0:
                        scale = bench_vol / realized_vol
                        port_ret *= scale
                weight_dict = {f'w_{col}': (weights[top_etfs.index(col)] if col in top_etfs else 0.0) for col in price_df.columns}
                results.append({
                    'date': dates[i + 1],
                    'port_return': port_ret,
                    'sharpe': np.nan,
                    'risk': np.nan,
                    'ret': np.nan,
                    **weight_dict
                })
            except Exception as e:
                results.append({
                    'date': dates[i + 1],
                    'port_return': np.nan,
                    'sharpe': np.nan,
                    'risk': np.nan,
                    'ret': np.nan,
                    **{f'w_{col}': np.nan for col in price_df.columns}
                })
    df = pd.DataFrame(results)
    return df.set_index('date')

def grid_search_alpha_params(prices, macro_factors, top_n=1, long_short=False):
    """
    Grid search over momentum/value lookbacks and blend weights to maximize Sharpe.
    Prints the best Sharpe and corresponding parameters.
    """
    best_sharpe = -np.inf
    best_params = None
    results = []
    for mom_lb in [63, 126, 252]:
        for val_lb in [126, 252, 504]:
            for mom_w in [1.0, 0.75, 0.5, 0.25, 0.0]:
                val_w = 1.0 - mom_w
                alpha_signals = compute_alpha_signals(
                    prices,
                    momentum_lookback=mom_lb,
                    value_lookback=val_lb,
                    momentum_weight=mom_w,
                    value_weight=val_w
                )
                result = rolling_alpha_backtest(
                    prices, window=252, min_risk=0.03, long_short=long_short, top_n=top_n,
                    target_vol=True, benchmark_returns=macro_factors['MKT'],
                    alpha_signals=alpha_signals,
                    momentum_lookback=mom_lb,
                    value_lookback=val_lb,
                    momentum_weight=mom_w,
                    value_weight=val_w
                )
                port_ret = result['port_return']
                sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)
                results.append((sharpe, mom_lb, val_lb, mom_w, val_w))
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = result
                    best_params = (mom_lb, val_lb, mom_w, val_w)
    print("\n=== Grid Search Results ===")
    for sharpe, mom_lb, val_lb, mom_w, val_w in sorted(results, reverse=True):
        print(f"Sharpe: {sharpe:.2f} | Momentum LB: {mom_lb} | Value LB: {val_lb} | Mom Wt: {mom_w:.2f} | Val Wt: {val_w:.2f}")
    print(f"Best Sharpe: {best_sharpe:.2f} with Momentum LB: {best_params[0]}, Value LB: {best_params[1]}, Mom Wt: {best_params[2]:.2f}, Val Wt: {best_params[3]:.2f}\n")
    return best_result, best_params  # <-- Return both

def plot_cumulative_returns(portfolio_returns, benchmark_returns, title="Cumulative Returns", filename=None):
    """
    Plot cumulative returns of portfolio and benchmark.
    Optionally save the plot to a file if filename is provided.
    """
    cum_port = (1 + portfolio_returns.dropna()).cumprod()
    cum_bench = (1 + benchmark_returns.reindex(cum_port.index).dropna()).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(cum_port, label="Portfolio")
    plt.plot(cum_bench, label="SPY Benchmark")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def plot_rolling_sharpe(portfolio_returns, benchmark_returns, window=126, title="Rolling Sharpe Ratio", filename=None):
    """
    Plot rolling Sharpe ratio for portfolio and benchmark.
    """
    roll_sharpe_port = portfolio_returns.rolling(window).mean() / portfolio_returns.rolling(window).std() * np.sqrt(252)
    roll_sharpe_bench = benchmark_returns.reindex(portfolio_returns.index).rolling(window).mean() / benchmark_returns.reindex(portfolio_returns.index).rolling(window).std() * np.sqrt(252)
    plt.figure(figsize=(10, 6))
    plt.plot(roll_sharpe_port, label="Portfolio")
    plt.plot(roll_sharpe_bench, label="SPY Benchmark")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def summary_stats(portfolio_returns, benchmark_returns):
    """
    Return a DataFrame with annualized return, volatility, Sharpe, and max drawdown.
    """
    ann_factor = 252
    stats = {}
    for name, ret in [("Portfolio", portfolio_returns), ("Benchmark", benchmark_returns.reindex(portfolio_returns.index))]:
        ret = ret.dropna()
        ann_ret = (1 + ret).prod() ** (ann_factor / len(ret)) - 1
        ann_vol = ret.std() * np.sqrt(ann_factor)
        sharpe = ret.mean() / ret.std() * np.sqrt(ann_factor)
        cum = (1 + ret).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        stats[name] = [ann_ret, ann_vol, sharpe, max_dd]
    return pd.DataFrame(stats, index=["Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown"]).T

if __name__ == "__main__":
    # Redirect all print output to results.txt
    with open("results.txt", "w") as f, \
         open("results.txt", "a") as fa:  # 'a' for appending after initial write
        sys.stdout = f

        # 1. Define ETF universe (restrict to US equity ETFs for better SPY comparison)
        etf_universe = [
            'SPY', 'IVV', 'VOO', 'VTI', 'XLK', 'XLF', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLU'
        ]
        start_date = "2020-01-01"
        end_date = "2025-01-01"

        # 2. Download price data
        print("Downloading ETF price data...")
        prices = yf.download(etf_universe, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
        prices = prices.dropna(axis=1, how='any')  # Drop ETFs with missing data

        # Download macro factors for benchmark returns
        macro_factors = download_factor_returns(start_date, end_date)

        # 3. Test rolling backtest (long-only, top 3 ETFs, volatility targeting)
        print("Running rolling alpha backtest testing (long-only, momentum only, top 3 US equity ETFs, volatility targeting)...")
        result_long = rolling_alpha_backtest(
            prices, window=252, min_risk=0.03, long_short=False, top_n=5,
            target_vol=True, benchmark_returns=macro_factors['MKT'],
            transaction_cost=0.0005  # 5bps per unit turnover
        )
        print("Long-only backtest testing completed successfully.")

        # 4. Test rolling backtest (long-short, top 3 ETFs, volatility targeting)
        print("Running rolling alpha backtest testing (long-short, momentum only, top 3 US equity ETFs, volatility targeting)...")
        result_longshort = rolling_alpha_backtest(
            prices, window=252, min_risk=0.03, long_short=True, top_n=3,
            target_vol=True, benchmark_returns=macro_factors['MKT']
        )
        print("Long-short backtest testing completed successfully.")

        # 5. Print summary stats and Sharpe ratio comparison
        macro_factors = download_factor_returns(result_long.index.min(), result_long.index.max())

        # 6. Grid search for optimal alpha parameters
        print("\n=== Grid Search for Optimal Alpha Parameters ===")
        result_long, _ = grid_search_alpha_params(prices, macro_factors, top_n=1, long_short=False)
        result_long.to_csv("rolling_alpha_results_long.csv")
        print("Long-only backtest complete. Results saved to rolling_alpha_results_long.csv.")
        
        result_longshort, _ = grid_search_alpha_params(prices, macro_factors, top_n=1, long_short=True)
        result_longshort.to_csv("rolling_alpha_results_longshort.csv")
        print("Long-short backtest complete. Results saved to rolling_alpha_results_longshort.csv.")

        # 7. Plot cumulative returns and print summary stats
        plot_cumulative_returns(result_long['port_return'], macro_factors['MKT'], title="Long-Only Portfolio vs. SPY", filename="long_only_vs_spy.png")
        plot_cumulative_returns(result_longshort['port_return'], macro_factors['MKT'], title="Long-Short Portfolio vs. SPY", filename="long_short_vs_spy.png")
        plot_rolling_sharpe(result_long['port_return'], macro_factors['MKT'], title="Rolling Sharpe Ratio (Long-Only)", filename="rolling_sharpe_long.png")
        plot_rolling_sharpe(result_longshort['port_return'], macro_factors['MKT'], title="Rolling Sharpe Ratio (Long-Short)", filename="rolling_sharpe_longshort.png")

        print("\n=== Summary Stats (Long-Only) ===")
        print(summary_stats(result_long['port_return'], macro_factors['MKT']))

        print("\n=== Summary Stats (Long-Short) ===")
        print(summary_stats(result_longshort['port_return'], macro_factors['MKT']))

        # 8. Performance attribution using macro factors
        print("\n=== Performance Attribution: Long-Only Portfolio ===")
        performance_attribution(result_long['port_return'], macro_factors)

        print("\n=== Performance Attribution: Long-Short Portfolio ===")
        performance_attribution(result_longshort['port_return'], macro_factors)

        print("\n=== Fama-French 3-Factor Attribution: Long-Only Portfolio ===")
        advanced_performance_attribution(result_long['port_return'], result_long.index.min(), result_long.index.max(), use_ff5=False)
        print("\n=== Fama-French 3-Factor Attribution: Long-Short Portfolio ===")
        advanced_performance_attribution(result_longshort['port_return'], result_longshort.index.min(), result_longshort.index.max(), use_ff5=False)

        # Split at 2024-01-01
        split_date = "2024-01-01"
        train_prices = prices.loc[:split_date]
        test_prices = prices.loc[split_date:]
        train_macro = macro_factors.loc[:split_date]
        test_macro = macro_factors.loc[split_date:]

        # Grid search on training set
        best_train_result, best_params = grid_search_alpha_params(train_prices, train_macro, top_n=3, long_short=False)
        best_mom_lb, best_val_lb, best_mom_w, best_val_w = best_params

        # Use best parameters to run test set backtest
        alpha_signals_test = compute_alpha_signals(
            test_prices,
            momentum_lookback=best_mom_lb,
            value_lookback=best_val_lb,
            momentum_weight=best_mom_w,
            value_weight=best_val_w
        )

        # Determine the rolling window for the test set
        test_window = min(5, len(test_prices)-1)
        if test_window < 1:
            print(f"Skipping walk-forward step: test_window={test_window} < 1")
        else:
            test_result = rolling_alpha_backtest(
                test_prices, window=test_window, min_risk=0.03, long_short=False, top_n=3,
                target_vol=True, benchmark_returns=test_macro['MKT'],
                alpha_signals=alpha_signals_test,
                momentum_lookback=best_mom_lb,
                value_lookback=best_val_lb,
                momentum_weight=best_mom_w,
                value_weight=best_val_w
            )

        # Walk-forward: re-optimize parameters every N months
        window_size = 504  # e.g., 2 years
        step_size = 21     # e.g., 1 month
        walkforward_results_long = []
        walkforward_results_longshort = []

        for start_idx in range(0, len(prices) - window_size - step_size, step_size):
            # Slices for this step
            train_slice = prices.iloc[start_idx:start_idx+window_size]
            macro_slice = macro_factors.iloc[start_idx:start_idx+window_size]
            test_slice = prices.iloc[start_idx+window_size:start_idx+window_size+step_size]
            test_macro_slice = macro_factors.iloc[start_idx+window_size:start_idx+window_size+step_size]

            print(f"\nDEBUG: Walk-forward step {start_idx}")
            print("  train_slice shape:", train_slice.shape)
            print("  test_slice shape:", test_slice.shape)
            print("  test_macro_slice shape:", test_macro_slice.shape)

            # --- Long-Only Grid Search on Training Window ---
            best_train_result_long, best_params_long = grid_search_alpha_params(train_slice, macro_slice, top_n=3, long_short=False)
            best_mom_lb_long, best_val_lb_long, best_mom_w_long, best_val_w_long = best_params_long

            # Compute alpha signals for test window (long-only) using full price history
            alpha_signals_full_long = compute_alpha_signals(
                prices,
                momentum_lookback=best_mom_lb_long,
                value_lookback=best_val_lb_long,
                momentum_weight=best_mom_w_long,
                value_weight=best_val_w_long
            )
            alpha_signals_test_long = alpha_signals_full_long.loc[test_slice.index]
            print("  alpha_signals_test_long shape:", alpha_signals_test_long.shape)
            print("  alpha_signals_test_long NaN count:", alpha_signals_test_long.isna().sum().sum())

            rolling_window = min(5, len(test_slice)-1)
            if rolling_window < 1:
                print(f"  Skipping walk-forward step: test_window={rolling_window} < 1")
            else:
                # Backtest on test window (long-only)
                test_result_long = rolling_alpha_backtest(
                    test_slice, window=rolling_window, min_risk=0.03, long_short=False, top_n=3,
                    target_vol=True, benchmark_returns=test_macro_slice['MKT'] if 'MKT' in test_macro_slice else None,
                    alpha_signals=alpha_signals_test_long,
                    momentum_lookback=best_mom_lb_long,
                    value_lookback=best_val_lb_long,
                    momentum_weight=best_mom_w_long,
                    value_weight=best_val_w_long
                )
                print("  test_result_long shape:", test_result_long.shape)
                print("  test_result_long['port_return'] NaN count:", test_result_long['port_return'].isna().sum())
                walkforward_results_long.append(test_result_long)

            # --- Long-Short Grid Search on Training Window ---
            best_train_result_ls, best_params_ls = grid_search_alpha_params(train_slice, macro_slice, top_n=3, long_short=True)
            best_mom_lb_ls, best_val_lb_ls, best_mom_w_ls, best_val_w_ls = best_params_ls

            # Compute alpha signals for test window (long-short) using full price history
            alpha_signals_full_ls = compute_alpha_signals(
                prices,
                momentum_lookback=best_mom_lb_ls,
                value_lookback=best_val_lb_ls,
                momentum_weight=best_mom_w_ls,
                value_weight=best_val_w_ls
            )
            alpha_signals_test_ls = alpha_signals_full_ls.loc[test_slice.index]
            print("  alpha_signals_test_ls shape:", alpha_signals_test_ls.shape)
            print("  alpha_signals_test_ls NaN count:", alpha_signals_test_ls.isna().sum().sum())

            rolling_window = min(5, len(test_slice)-1)
            if rolling_window < 1:
                print(f"  Skipping walk-forward step (long-short): test_window={rolling_window} < 1")
            else:
                # Backtest on test window (long-short)
                test_result_ls = rolling_alpha_backtest(
                    test_slice, window=rolling_window, min_risk=0.03, long_short=True, top_n=3,
                    target_vol=True, benchmark_returns=test_macro_slice['MKT'] if 'MKT' in test_macro_slice else None,
                    alpha_signals=alpha_signals_test_ls,
                    momentum_lookback=best_mom_lb_ls,
                    value_lookback=best_val_lb_ls,
                    momentum_weight=best_mom_w_ls,
                    value_weight=best_val_w_ls
                )
                print("  test_result_ls shape:", test_result_ls.shape)
                print("  test_result_ls['port_return'] NaN count:", test_result_ls['port_return'].isna().sum())
                walkforward_results_longshort.append(test_result_ls)

        # Concatenate all walk-forward results for full performance series (long-only)
        if walkforward_results_long:
            walkforward_df_long = pd.concat(walkforward_results_long)
            print("DEBUG: walkforward_df_long shape:", walkforward_df_long.shape)
            print("DEBUG: walkforward_df_long head:\n", walkforward_df_long.head())
            print("DEBUG: walkforward_df_long['port_return'] NaN count:", walkforward_df_long['port_return'].isna().sum())
            walkforward_df_long = walkforward_df_long[~walkforward_df_long.index.duplicated(keep='first')]
            walkforward_df_long.sort_index(inplace=True)
            walkforward_df_long.to_csv("walkforward_results_long.csv")
            print("DEBUG: walkforward_df_long after dedup/sort shape:", walkforward_df_long.shape)
            print("DEBUG: walkforward_df_long['port_return'] after dedup/sort NaN count:", walkforward_df_long['port_return'].isna().sum())
            print("DEBUG: macro_factors['MKT'] shape:", macro_factors['MKT'].shape)
            print("DEBUG: macro_factors['MKT'] head:\n", macro_factors['MKT'].head())
            print("DEBUG: macro_factors['MKT'] NaN count:", macro_factors['MKT'].isna().sum())
            print("\n=== Walk-Forward Out-of-Sample Performance (Long-Only) ===")
            print(summary_stats(walkforward_df_long['port_return'], macro_factors['MKT']))
            plot_cumulative_returns(walkforward_df_long['port_return'], macro_factors['MKT'], title="Walk-Forward Long-Only Portfolio vs. SPY", filename="walkforward_long_only_vs_spy.png")
            plot_rolling_sharpe(walkforward_df_long['port_return'], macro_factors['MKT'], title="Walk-Forward Rolling Sharpe Ratio (Long-Only)", filename="walkforward_rolling_sharpe_long.png")
        else:
            print("No walk-forward results generated (long-only).")

        # Concatenate all walk-forward results for full performance series (long-short)
        if walkforward_results_longshort:
            walkforward_df_longshort = pd.concat(walkforward_results_longshort)
            print("DEBUG: walkforward_df_longshort shape:", walkforward_df_longshort.shape)
            print("DEBUG: walkforward_df_longshort head:\n", walkforward_df_longshort.head())
            print("DEBUG: walkforward_df_longshort['port_return'] NaN count:", walkforward_df_longshort['port_return'].isna().sum())
            walkforward_df_longshort = walkforward_df_longshort[~walkforward_df_longshort.index.duplicated(keep='first')]
            walkforward_df_longshort.sort_index(inplace=True)
            walkforward_df_longshort.to_csv("walkforward_results_longshort.csv")
            print("DEBUG: walkforward_df_longshort after dedup/sort shape:", walkforward_df_longshort.shape)
            print("DEBUG: walkforward_df_longshort['port_return'] after dedup/sort NaN count:", walkforward_df_longshort['port_return'].isna().sum())
            print("DEBUG: macro_factors['MKT'] shape:", macro_factors['MKT'].shape)
            print("DEBUG: macro_factors['MKT'] head:\n", macro_factors['MKT'].head())
            print("DEBUG: macro_factors['MKT'] NaN count:", macro_factors['MKT'].isna().sum())
            print("\n=== Walk-Forward Out-of-Sample Performance (Long-Short) ===")
            print(summary_stats(walkforward_df_longshort['port_return'], macro_factors['MKT']))
            plot_cumulative_returns(walkforward_df_longshort['port_return'], macro_factors['MKT'], title="Walk-Forward Long-Short Portfolio vs. SPY", filename="walkforward_longshort_vs_spy.png")
            plot_rolling_sharpe(walkforward_df_longshort['port_return'], macro_factors['MKT'], title="Walk-Forward Rolling Sharpe Ratio (Long-Short)", filename="walkforward_rolling_sharpe_longshort.png")
        else:
            print("No walk-forward results generated (long-short).")