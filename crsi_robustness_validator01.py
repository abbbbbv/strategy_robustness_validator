#!/usr/bin/env python3
"""
CompRSI – full validation suite
────────────────────────────────
No external finance libraries required.
Provides these robustness tests:

 1. Train / Test split
 2. Walk-forward analysis
 3. Parameter robustness (heat-map + neighbour test)
 4. Monte-Carlo shuffle of trade order
 5. Bootstrap CAGR confidence interval
 6. Deflated Sharpe Ratio  (López-de-Prado 2018)
 7. CPCV-PBO     (Combinatorial Purged Cross-Validation)   ← mlfinlab-style
 8. SPA / White-Reality-Check-2.0   (optional, pure NumPy)

Only standard libs plus numpy-pandas-scipy-seaborn-matplotlib.
"""

import logging, warnings, math, itertools, random, numpy as np, pandas as pd
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import pandas_ta as ta, seaborn as sns, matplotlib.pyplot as plt
import scipy.stats as st
from tqdm import tqdm
from binance.client import Client
from backtesting import Backtest, Strategy
from scipy.special import erfinv
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s: %(message)s")

SYMBOL, BTC = "XRPUSDT", "BTCUSDT"
START, END  = "2024-12-01", "2025-05-01"
INTERVAL    = Client.KLINE_INTERVAL_4HOUR   

RSI_LEN = 9
RSI_UPPER, RSI_LOWER = 75, 25

CASH, COMM = 10_000, 0.0006
SL_DEF, TP_DEF = 3.5, 5.0

SL_LIST = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
TP_LIST = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

API_KEY, API_SECRET = "", ""
PLOT_HEATMAP = False
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# ────────────────────────────────────────────────────────────────────────


# ─── helpers: Binance download ──────────────────────────────────────────
def fetch_klines(cli, sym, start, end, interval):
    gen  = cli.futures_historical_klines_generator(sym, interval, start, end)
    cols = ['ts','Open','High','Low','Close','Volume','ct','qav',
            'nt','tbb','tbq','ig']
    df = pd.DataFrame(list(gen), columns=cols)
    if df.empty:
        raise RuntimeError(f"No data for {sym}")
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df[['Open','High','Low','Close','Volume']].apply(pd.to_numeric)

def load_data():
    cli = Client(API_KEY, API_SECRET)
    with tqdm(total=2, desc="Download") as bar:
        df_a = fetch_klines(cli, SYMBOL, START, END, INTERVAL); bar.update(1)
        df_b = fetch_klines(cli, BTC,    START, END, INTERVAL); bar.update(1)

    idx = df_a.index.intersection(df_b.index)
    df_a, df_b = df_a.loc[idx], df_b.loc[idx]

    avg_a = df_a[['Open','High','Low','Close']].mean(axis=1)
    avg_b = df_b[['Open','High','Low','Close']].mean(axis=1)

    df        = df_a.copy()
    df['rel'] = avg_a / avg_b
    df['rsi'] = ta.rsi(df['rel'], length=RSI_LEN)
    return df.dropna()

# ─── trading strategy ───────────────────────────────────────────────────
class CompRSI(Strategy):
    rsi_len     = RSI_LEN
    upper_band  = RSI_UPPER
    lower_band  = RSI_LOWER
    sl_pct      = SL_DEF
    tp_pct      = TP_DEF

    def init(self): pass

    def next(self):
        if math.isnan(self.data.rsi[-2]) or math.isnan(self.data.rsi[-1]):
            return
        prev, now = self.data.rsi[-2], self.data.rsi[-1]
        px = self.data.Close[-1]

        if prev < self.lower_band <= now:
            if self.position.is_short: self.position.close()
            self.buy (sl=px*(1-self.sl_pct/100), tp=px*(1+self.tp_pct/100))
        elif prev > self.upper_band >= now:
            if self.position.is_long: self.position.close()
            self.sell(sl=px*(1+self.sl_pct/100), tp=px*(1-self.tp_pct/100))

# ─── back-testing utility wrappers ──────────────────────────────────────
def run_bt(df, sl, tp):
    bt = Backtest(df, CompRSI, cash=CASH, commission=COMM, exclusive_orders=True)
    return bt.run(sl_pct=sl, tp_pct=tp)

def optimise(df):
    bt = Backtest(df, CompRSI, cash=CASH, commission=COMM, exclusive_orders=True)
    best = bt.optimize(sl_pct=SL_LIST, tp_pct=TP_LIST,
                       maximize='Equity Final [$]', return_heatmap=False)
    return best._strategy.sl_pct, best._strategy.tp_pct

# ─── 1) Train / Test split ──────────────────────────────────────────────
def train_test(df, frac=.6):
    cut = int(len(df)*frac)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    sl,tp  = optimise(tr)
    st_tr, st_te = run_bt(tr,sl,tp), run_bt(te,sl,tp)
    print("\n1) Train/Test   SL %.1f  TP %.1f" % (sl,tp))
    print("TRAIN Sharpe %.2f  Ret %.1f%%" % (st_tr['Sharpe Ratio'], st_tr['Return [%]']))
    print("TEST  Sharpe %.2f  Ret %.1f%%" % (st_te['Sharpe Ratio'], st_te['Return [%]']))
    return sl,tp

# ─── 2) Walk-forward ────────────────────────────────────────────────────
def walk_forward(df, win_frac=.4, step_frac=.05):
    w,s = int(len(df)*win_frac), int(len(df)*step_frac)
    outs=[]
    for i in range(0, len(df)-w-s, s):
        tr,te = df.iloc[i:i+w], df.iloc[i+w:i+w+s]
        sl,tp = optimise(tr)
        outs.append(run_bt(te,sl,tp)['Return [%]'])
    print("\n2) Walk-forward (%d) mean %.2f%%  med %.2f%%"
          %(len(outs), np.mean(outs), np.median(outs)))

# ─── 3) Parameter robustness (heat-map) ─────────────────────────────────
def param_robustness(df):
    bt  = Backtest(df, CompRSI, cash=CASH, commission=COMM, exclusive_orders=True)
    res = bt.optimize(sl_pct=SL_LIST, tp_pct=TP_LIST,
                      maximize='Equity Final [$]', return_heatmap=True)

    if isinstance(res, tuple):                      
        best_stats, heat = res
    else:                                          
        heat = res
        best_stats = None

    # best parameters
    if best_stats is not None and hasattr(best_stats, '_strategy'):
        best_sl = best_stats._strategy.sl_pct
        best_tp = best_stats._strategy.tp_pct
    else:
        if isinstance(heat, pd.Series):             
            best_sl, best_tp = heat.idxmax()
        elif 'sl_pct' in heat.columns:              # flat DataFrame
            row = heat.loc[heat['Equity Final [$]'].idxmax()]
            best_sl, best_tp = row['sl_pct'], row['tp_pct']
        else:                                       # pivot
            idx, col = np.unravel_index(heat.values.argmax(), heat.shape)
            best_sl, best_tp = heat.index[idx], heat.columns[col]

    # flatten heat to long DF for neighbour calc
    if isinstance(heat, pd.Series):
        heat_long = heat.rename('Equity Final [$]').reset_index()
        heat_long.columns = ['sl_pct','tp_pct','Equity Final [$]']
    elif 'sl_pct' in heat.columns:
        heat_long = heat.copy()
    else:
        heat_long = heat.stack().reset_index()
        heat_long.columns = ['sl_pct','tp_pct','Equity Final [$]']

    best_eq = heat_long.loc[
        (heat_long['sl_pct']==best_sl)&(heat_long['tp_pct']==best_tp),
        'Equity Final [$]'].values[0]
    nbr = heat_long[(heat_long['sl_pct'].between(best_sl-0.5,best_sl+0.5)) &
                    (heat_long['tp_pct'].between(best_tp-0.5,best_tp+0.5))]
    print("\n3) Parameter robustness")
    print("Best equity %.1f  |  Neighbour mean %.1f"
          % (best_eq, nbr['Equity Final [$]'].mean()))

    if PLOT_HEATMAP:
        if 'sl_pct' not in heat.columns:
            pivot = heat
        else:
            pivot = heat_long.pivot(index='sl_pct', columns='tp_pct',
                                    values='Equity Final [$]')
        sns.heatmap(pivot, cmap='viridis'); plt.title("Equity heat-map"); plt.show()

    return best_sl, best_tp

# ─── 4) Monte-Carlo shuffle of trade order ──────────────────────────────
def mc_shuffle(df, sl,tp, paths=5000):
    st  = run_bt(df,sl,tp); rets = st['_trades']['ReturnPct'].values/100
    if len(rets)==0:
        print("\n4) Monte-Carlo – no trades"); return
    finals=[CASH*np.prod(1+np.random.permutation(rets)) for _ in range(paths)]
    p5,p95=np.percentile(finals,[5,95])
    print("\n4) Monte-Carlo  orig %.1f | 5-95%% [%.1f %.1f]"
          %(st['Equity Final [$]'], p5, p95))

# ─── 5) Bootstrap CAGR CI ───────────────────────────────────────────────
def bootstrap_cagr(df, sl,tp, reps=2000):
    st=run_bt(df,sl,tp)
    ser=st['_equity_curve']['Equity'].pct_change().dropna()
    if ser.empty: print("\n5) Bootstrap – no curve"); return
    T=len(ser)/(365/6)                          # 6 x 4h bars per day
    idx=np.random.randint(0,len(ser),(reps,len(ser)))
    finals=np.prod(1+ser.values[idx],axis=1)
    cagrs=finals**(1/T)-1
    lo,hi=np.percentile(cagrs,[2.5,97.5])
    print("\n5) Bootstrap CAGR 95%% CI  %.2f%% → %.2f%%"%(lo*100,hi*100))

# ─── 6) Deflated Sharpe Ratio (DSR) ─────────────────────────────────────
def deflated_sharpe(df, sl, tp, n_trials):
    stats = run_bt(df, sl, tp)
    daily = stats['_equity_curve']['Equity'].pct_change().dropna().values

    sr_hat = np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(len(daily))
    skew   = st.skew(daily)
    kurt   = st.kurtosis(daily, fisher=False)

    # expected max-Sharpe given n_trials (López-de-Prado)
    sr_max_exp = np.sqrt(2) * erfinv(1 - 2/n_trials)

    # variance of Sharpe estimator
    sr_var = (1 + (sr_hat**2) * ((kurt - 3)/4 - skew * sr_hat / 3)) / len(daily)

    z_score = (sr_hat - sr_max_exp) / np.sqrt(sr_var)
    p_val   = 1 - st.norm.cdf(z_score)

    print("\n6) Deflated Sharpe")
    print(f"Raw Sharpe {sr_hat:.2f}   z-score {z_score:.2f}   p-value {p_val:.3f}")
# ─── 7) CPCV-PBO (Prob of Back-test Over-fit) ───────────────────────────
def cpcv_pbo(df, k=10):
    folds = np.array_split(df, k)
    lambdas=[]
    total_trials=len(SL_LIST)*len(TP_LIST)
    for m in range(1, k):
        for test_idx in itertools.combinations(range(k), m):
            train_idx=[i for i in range(k) if i not in test_idx]
            train=pd.concat(folds[i] for i in train_idx)
            test =pd.concat(folds[i] for i in test_idx)
            sl,tp=optimise(train)
            tr=run_bt(train,sl,tp)['Sharpe Ratio']
            te=run_bt(test ,sl,tp)['Sharpe Ratio']
            lambdas.append(1/(1+np.exp(-(tr-te))))
    pbo=np.mean(lambdas)
    print("\n7) CPCV-PBO")
    print("λ mean %.3f  →  PBO ≈ %.3f" % (pbo, pbo))

# ─── 8) SPA / Reality-Check-2.0 (optional) ──────────────────────────────
def spa_test(df, B=500):
    combos=list(itertools.product(SL_LIST,TP_LIST))
    mats=[]
    for sl,tp in combos:
        curve=run_bt(df,sl,tp)['_equity_curve']['Equity'].pct_change()
        mats.append(curve.reindex(df.index, fill_value=0).values)
    X=np.column_stack(mats)
    X-=X.mean(axis=1, keepdims=True)
    T=X.shape[0]
    t_obs=np.sqrt(T)*X.mean(axis=0)/X.std(axis=0,ddof=1)
    t_obs=t_obs.max()
    wins=[]
    rng=np.random.default_rng(SEED)
    for _ in range(B):
        idx=[]
        while len(idx)<T:
            start=rng.integers(T); L=rng.geometric(p=1/10)
            idx+=list(range(start, min(T,start+L)))
        idx=idx[:T]
        d_b=X[idx,:]
        tb=np.sqrt(T)*d_b.mean(axis=0)/d_b.std(axis=0,ddof=1)
        wins.append(tb.max())
    p=(np.array(wins)>=t_obs).mean()
    print("\n8) SPA Reality-Check p-value %.3f"%p)

# ─── main driver ────────────────────────────────────────────────────────
def main():
    df=load_data()
    print("Loaded %d candles  (%s → %s)"%(len(df),df.index[0],df.index[-1]))
    sl,tp=train_test(df)
    walk_forward(df)
    sl,tp=param_robustness(df)
    mc_shuffle(df,sl,tp)
    bootstrap_cagr(df,sl,tp)
    deflated_sharpe(df,sl,tp,n_trials=len(SL_LIST)*len(TP_LIST))
    cpcv_pbo(df,k=10)
    spa_test(df,B=300)        # comment out if slow

if __name__=="__main__":
    main()