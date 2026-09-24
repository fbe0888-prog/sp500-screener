#!/usr/bin/env python3
"""
S&P 500 Multi-Edge Screener — Cloud Version v6
================================================
الإصلاحات والإضافات:
- A: تنظيف تلقائي للصفوف التالفة في picks_history
- B: إرسال إيميل HTML بأفضل 3 أسهم + إحصاءات
- C: رسم بياني لـHit Rate داخل Google Sheets
- إصلاحات سابقة: NaN handling, Timestamp serialization, دفاعية كاملة
"""

import os
import sys
import time
import warnings
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

warnings.filterwarnings("ignore")

# ============================================================
# الإعدادات
# ============================================================
SHEET_ID = os.environ.get("SHEET_ID")
SA_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "sa.json")
PERIOD = "1y"
HOLD_DAYS = 5
TOP_N = 8
MIN_SCORE = 12.0
JACCARD_THRESHOLD = 0.70
BATCH_SIZE = 50
MAX_META = None

# ============================================================
# الإيدجات
# ============================================================
EDGE_LABELS = {
    21: "20-day breakout", 22: "55-day breakout", 23: "Multi-unit trail (vol stop)",
    24: "Aligned w/ long-term trend", 25: "Stage 2 advance", 26: "Darvas box breakout+vol",
    27: "Livermore pivotal point", 28: "Riding established trend", 29: "Above 30-week MA",
    30: "Relative strength vs SPY",
    31: "Qtr earnings accel", 32: "Annual earnings growth", 33: "New price highs",
    34: "Low float / supply-demand", 35: "Industry leader (RS rank)",
    36: "Institutional sponsorship proxy", 37: "Favorable market direction",
    38: "Breakout from proper base", 39: "Volume confirms breakout", 40: "Top RS ranking",
    41: "Turtle Soup (failed 20d breakout)", 42: "80-20 pattern",
    43: "Momentum Pinball (RSI+MA)", 44: "Holy Grail (ADX pullback)",
    45: "NR7 / opening-range setup", 46: "Climax reversal",
    47: "2-period RSI extreme", 48: "Consecutive-day reversal setup",
    49: "Gap fill / gap-and-go", 50: "Support/resistance test",
    51: "Wyckoff spring/upthrust", 52: "Volume climax / dry-up",
    53: "Effort vs result divergence", 54: "Smart-money footprint proxy",
    55: "Prior swing high/low S/R", 56: "Price at key MA",
    57: "Regime-suited setup", 58: "Low-ADX caution flag",
    59: "Vol+trend aligned aggression", 60: "Breadth/intermarket context",
}

GATE_EDGES = {23, 24, 29, 37}
DISPLAY_EXCLUDE = {23, 24, 29, 31, 32, 37, 57}

DEFAULT_WEIGHTS = {e: 1.0 for e in EDGE_LABELS}
for e in GATE_EDGES:
    DEFAULT_WEIGHTS[e] = 0.25
for e in (26, 38, 39, 44, 51):
    DEFAULT_WEIGHTS[e] = 1.8
for e in (21, 22, 27, 33, 42, 47, 45, 46):
    DEFAULT_WEIGHTS[e] = 1.5
for e in (41, 43, 48, 49, 50):
    DEFAULT_WEIGHTS[e] = 1.3
for e in (25, 28, 40):
    DEFAULT_WEIGHTS[e] = 1.1


@dataclass
class FilterConfig:
    min_price: float = 15.0
    min_avg_vol_30d: float = 1_000_000
    min_market_cap: float = 10_000_000_000
    exclude_earnings_window: bool = True


# ============================================================
# المؤشرات الفنية
# ============================================================
def true_range(df):
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df, n=20):
    return true_range(df).rolling(n).mean()


def rsi(series, n=2):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx(df, n=14):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_n = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(n).mean(), plus_di, minus_di


def sma(series, n):
    return series.rolling(n).mean()


def is_nr7(df):
    rng = df["high"] - df["low"]
    return rng == rng.rolling(7).min()


def compute_features(df):
    df = df.sort_values("date").reset_index(drop=True)
    df["sma20"] = sma(df["close"], 20)
    df["sma50"] = sma(df["close"], 50)
    df["sma150"] = sma(df["close"], min(150, max(20, len(df) - 1)))
    df["atr20"] = atr(df, 20)
    df["adx14"], df["plus_di"], df["minus_di"] = adx(df, 14)
    df["rsi2"] = rsi(df["close"], 2)
    df["hh20"] = df["high"].rolling(20).max()
    df["ll20"] = df["low"].rolling(20).min()
    df["hh55"] = df["high"].rolling(55).max()
    df["ll55"] = df["low"].rolling(55).min()
    df["avg_vol_30d"] = df["volume"].rolling(30).mean()
    df["avg_vol_50d"] = df["volume"].rolling(50).mean()
    df["nr7"] = is_nr7(df)
    df["ret1"] = df["close"].pct_change()
    df["up_streak"] = (df["ret1"] > 0).astype(int).groupby((df["ret1"] <= 0).cumsum()).cumsum()
    df["down_streak"] = (df["ret1"] < 0).astype(int).groupby((df["ret1"] >= 0).cumsum()).cumsum()
    df["range_pct_of_atr"] = (df["high"] - df["low"]) / df["atr20"]
    df["vol_ratio"] = df["volume"] / df["avg_vol_50d"]
    return df


# ============================================================
# score_ticker
# ============================================================
@dataclass
class ScoreResult:
    ticker: str
    score: float
    active_edges: list = field(default_factory=list)
    top_edges: list = field(default_factory=list)
    style_votes: dict = field(default_factory=dict)
    entry: float = None
    stop: float = None
    target: float = None
    reason: str = ""
    regime_fit: str = "neutral"
    confidence: str = "low"


def score_ticker(df, ticker, meta_row, market_trend_up, regime, weights):
    if len(df) < 60:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    active = []
    pts = 0.0

    def fire(edge_num, cond, w=None):
        nonlocal pts
        if cond:
            if edge_num not in active:
                active.append(edge_num)
            pts += weights.get(edge_num, 1.0) if w is None else w

    trend_up = last["close"] > last["sma50"] > (df["sma150"].iloc[-1]
                if not np.isnan(df["sma150"].iloc[-1]) else -np.inf)

    fire(21, last["close"] >= last["hh20"] * 0.999)
    fire(22, last["close"] >= last["hh55"] * 0.999)
    fire(23, not np.isnan(last["atr20"]) and last["atr20"] > 0)
    fire(24, trend_up)
    fire(25, last["close"] > last["sma50"] and last["sma50"] > df["sma50"].iloc[-10]
         if len(df) > 60 else False)
    fire(26, (last["close"] >= last["hh20"] * 0.999) and last["vol_ratio"] > 1.3)
    fire(27, last["close"] > df["close"].iloc[-20:-1].max() and last["vol_ratio"] > 1.2)
    fire(28, trend_up and (last["adx14"] > 20 if not np.isnan(last["adx14"]) else False))
    fire(29, last["close"] > last["sma150"] if not np.isnan(last["sma150"]) else False)

    fire(33, last["close"] >= df["close"].rolling(252).max().iloc[-1] * 0.98
         if len(df) > 252 else last["close"] >= last["hh55"] * 0.98)
    fire(37, market_trend_up)
    fire(38, (last["close"] >= last["hh55"] * 0.98) and
         ((df["close"].iloc[-40:-5].max() - df["close"].iloc[-40:-5].min())
          / df["close"].iloc[-40:-5].mean() < 0.20) if len(df) > 45 else False)
    fire(39, (last["close"] >= last["hh20"] * 0.999) and last["vol_ratio"] > 1.3)

    broke_20 = (prev["high"] > df["hh20"].iloc[-3]) if len(df) > 3 else False
    fire(41, broke_20 and last["close"] < prev["close"])
    fire(42, last["rsi2"] < 10 and trend_up)
    fire(43, last["rsi2"] < 10 and last["close"] > last["sma50"])
    fire(44, (not np.isnan(last["adx14"])) and last["adx14"] > 25
         and last["close"] < last["sma20"] and trend_up)
    fire(45, bool(last["nr7"]))

    rsi2_val = last["rsi2"]
    if not np.isnan(rsi2_val):
        if rsi2_val < 2 or rsi2_val > 98:
            fire(47, True, w=weights.get(47, 1.5) * 1.5)
        elif rsi2_val < 5 or rsi2_val > 95:
            fire(47, True, w=weights.get(47, 1.5) * 0.5)

    near_hh = abs(last["close"] - last["hh20"]) / last["hh20"] < 0.01
    near_ll = abs(last["close"] - last["ll20"]) / last["ll20"] < 0.01
    fire(50, (near_hh or near_ll) and last["vol_ratio"] > 1.2)

    fire(46, last["vol_ratio"] > 2.0 and last["range_pct_of_atr"] > 1.5)
    fire(48, last["down_streak"] >= 4 or last["up_streak"] >= 4)
    gap = (last["open"] - prev["close"]) / prev["close"] if prev["close"] else 0
    fire(49, abs(gap) > 0.02)

    spring = last["low"] < last["ll20"] and last["close"] > df["ll20"].iloc[-2]
    fire(51, spring)
    fire(52, last["vol_ratio"] > 2.0 or last["vol_ratio"] < 0.5)
    fire(53, last["vol_ratio"] > 1.5 and abs(last["ret1"]) < 0.005)
    fire(54, last["vol_ratio"] > 1.5 and last["close"] > last["open"])
    fire(56, abs(last["close"] - last["sma50"]) / last["sma50"] < 0.01
         or abs(last["close"] - last["sma20"]) / last["sma20"] < 0.01)

    if meta_row is not None:
        q_growth = meta_row.get("eps_growth_qtr_pct")
        a_growth = meta_row.get("eps_growth_annual_pct")
        if pd.notna(q_growth):
            fire(31, q_growth > 0.20)
        if pd.notna(a_growth):
            fire(32, a_growth > 0.15)

    trend_set = {21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 37, 38, 39}
    mr_set = {41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52}
    trend_count = len(set(active) & trend_set)
    mr_count = len(set(active) & mr_set)

    if trend_count >= 3 and mr_count <= 1:
        style = "Trend"
    elif mr_count >= 3 and trend_count <= 1:
        style = "Mean-Reversion"
    elif trend_count > mr_count:
        style = "Trend"
    elif mr_count > trend_count:
        style = "Mean-Reversion"
    else:
        style = "Mixed"

    regime_fit = "neutral"
    if regime == "mean_reversion":
        if style == "Mean-Reversion":
            regime_fit = "aligned"
        elif style == "Trend":
            regime_fit = "against"
            pts *= 0.55
    elif regime == "trend":
        if style == "Trend":
            regime_fit = "aligned"
        elif style == "Mean-Reversion":
            regime_fit = "against"
            pts *= 0.55
    elif regime == "choppy":
        if style == "Mean-Reversion":
            regime_fit = "aligned"
        else:
            pts *= 0.75

    max_possible = sum(weights.get(e, 1.0) for e in weights) + 4
    normalized_score = max(0, min(100, (pts / max_possible) * 100))

    if normalized_score >= 25:
        confidence = "high"
    elif normalized_score >= 15:
        confidence = "medium"
    else:
        confidence = "low"

    entry = last["close"]
    atrv = last["atr20"] if not np.isnan(last["atr20"]) else last["close"] * 0.02
    if style == "Trend" or (style == "Mixed" and trend_count >= mr_count):
        stop = entry - 2 * atrv
        target = entry + 4 * atrv
    else:
        stop = entry - 1.5 * atrv
        target = entry + 3 * atrv

    sorted_active = sorted(set(active), key=lambda e: -weights.get(e, 1.0))
    top_edges = [e for e in sorted_active if e not in DISPLAY_EXCLUDE][:6]
    reason_bits = [EDGE_LABELS.get(e, str(e)) for e in top_edges[:3]]
    reason = f"{ticker}: {', '.join(reason_bits)}." if reason_bits else f"{ticker}: no distinctive edge."

    return ScoreResult(
        ticker=ticker,
        score=round(normalized_score, 1),
        active_edges=sorted(set(active)),
        top_edges=top_edges,
        style_votes={"trend": trend_count, "mean_reversion": mr_count},
        entry=round(entry, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        reason=reason,
        regime_fit=regime_fit,
        confidence=confidence,
    )


def detect_market_regime(spy_df):
    if spy_df is None or len(spy_df) < 60:
        return "unknown", True
    spy_df = compute_features(spy_df)
    last = spy_df.iloc[-1]
    market_trend_up = last["close"] > last["sma50"]
    if np.isnan(last["adx14"]):
        return "unknown", market_trend_up
    if last["adx14"] > 25:
        regime = "trend"
    elif last["adx14"] < 18:
        regime = "mean_reversion"
    else:
        regime = "choppy"
    return regime, market_trend_up


# ============================================================
# جلب البيانات
# ============================================================
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        tables = pd.read_html(url, storage_options=headers)
        sp500 = tables[0]
        tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"✅ {len(tickers)} سهم من Wikipedia")
        return tickers
    except Exception as e:
        print(f"⚠️ فشل جلب القائمة: {e}")
        return []


def download_prices(tickers, period="1y"):
    print(f"⏳ تحميل أسعار {len(tickers)} سهم...")
    all_data = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"   دفعة {i // BATCH_SIZE + 1}: {batch[0]}..{batch[-1]}")
        try:
            df = yf.download(batch, period=period, interval="1d",
                             group_by="ticker", auto_adjust=False,
                             progress=False, threads=True)
            all_data.append(df)
        except Exception as e:
            print(f"   ⚠️ فشل: {e}")
        time.sleep(1)
    if not all_data:
        return pd.DataFrame()
    combined = pd.concat(all_data, axis=1)
    print(f"✅ {combined.shape}")
    return combined


def prices_to_long(prices_wide):
    records = []
    for ticker in prices_wide.columns.levels[0]:
        try:
            sub = prices_wide[ticker].copy()
            sub.columns = [c.lower() for c in sub.columns]
            sub = sub.reset_index()
            sub = sub.rename(columns={"index": "date", "Date": "date",
                                       "Adj Close": "adj_close"})
            sub["ticker"] = ticker
            records.append(sub)
        except Exception:
            continue
    if not records:
        return pd.DataFrame()
    long_df = pd.concat(records, ignore_index=True)
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df = long_df.sort_values(["ticker", "date"])

    print("\n" + "=" * 60)
    print("🔍 التحقق من آخر صف لكل سهم")
    print("=" * 60)
    last_rows = long_df.groupby("ticker").tail(1)
    nan_close = last_rows["close"].isna().sum()
    total = len(last_rows)
    last_date_with_close = long_df.dropna(subset=["close"])["date"].max()
    last_date_any = long_df["date"].max()

    print(f"عدد الأسهم: {total}")
    print(f"آخر تاريخ في البيانات (أي صف): {last_date_any.date()}")
    print(f"آخر تاريخ فيه close مكتمل:    {last_date_with_close.date()}")
    print(f"أسهم بدون close في آخر صف:    {nan_close}/{total}")

    if nan_close > 0:
        sample_missing = last_rows[last_rows["close"].isna()]["ticker"].head(5).tolist()
        print(f"   ⚠️ أمثلة: {sample_missing}")

    long_df = long_df.dropna(subset=["close"])
    long_df = long_df.reset_index(drop=True)
    print(f"✅ Long: {long_df.shape[0]:,} صف بعد التنظيف")
    return long_df


def validate_latest_date(long_df):
    print("\n" + "=" * 60)
    print("🔍 التحقق من حداثة البيانات")
    print("=" * 60)
    latest_data_date = long_df["date"].max()
    today = pd.Timestamp.now().normalize()
    days_diff = (today - latest_data_date).days

    print(f"آخر تاريخ في البيانات:  {latest_data_date.date()}")
    print(f"اليوم:                  {today.date()}")
    print(f"الفارق:                 {days_diff} يوم")

    if days_diff > 5:
        print(f"⚠️ تحذير: البيانات قديمة بـ {days_diff} يوم")
    elif days_diff > 3:
        print(f"ℹ️ ملاحظة: بيانات قديمة قليلًا — طبيعي في عطلة نهاية الأسبوع")
    else:
        print(f"✅ البيانات حديثة")


def fetch_metadata(tickers, max_tickers=None):
    if max_tickers:
        tickers = tickers[:max_tickers]
    print(f"⏳ جلب الميتاداتا لـ {len(tickers)} سهم...")
    meta_rows = []
    for i, tkr in enumerate(tickers):
        try:
            t = yf.Ticker(tkr)
            info = t.info or {}
            next_earnings = None
            try:
                cal = t.calendar or {}
                if isinstance(cal, dict):
                    ne = cal.get("Earnings Date")
                    if isinstance(ne, list):
                        ne = ne[0] if ne else None
                    next_earnings = ne
            except Exception:
                pass
            meta_rows.append({
                "ticker": tkr,
                "market_cap": info.get("marketCap"),
                "next_earnings_date": next_earnings,
                "eps_growth_qtr_pct": info.get("earningsQuarterlyGrowth"),
                "eps_growth_annual_pct": info.get("earningsGrowth"),
                "sector": info.get("sector"),
            })
        except Exception:
            meta_rows.append({
                "ticker": tkr, "market_cap": None,
                "next_earnings_date": None, "eps_growth_qtr_pct": None,
                "eps_growth_annual_pct": None, "sector": None,
            })
        if (i + 1) % 100 == 0:
            print(f"   ... {i + 1}/{len(tickers)}")
        time.sleep(0.15)
    meta_df = pd.DataFrame(meta_rows)
    print(f"✅ {len(meta_df)} صف")
    return meta_df


def download_spy(period="1y"):
    print("⏳ تحميل SPY...")
    spy = yf.download("SPY", period=period, interval="1d",
                       auto_adjust=False, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    spy = spy.reset_index()
    spy.columns = [c.lower() for c in spy.columns]
    spy = spy.rename(columns={"adj close": "adj_close"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").reset_index(drop=True)
    spy = spy.dropna(subset=["close"])
    print(f"✅ SPY: {len(spy)} صف، آخر تاريخ: {spy['date'].max().date()}")
    return spy


def build_earnings_blackout(meta_df, window_days=3):
    blackout = set()
    if meta_df is None or "next_earnings_date" not in meta_df.columns:
        return blackout
    today = pd.Timestamp.now().normalize()
    for _, row in meta_df.iterrows():
        ned = row.get("next_earnings_date")
        if pd.isna(ned) or ned is None:
            continue
        try:
            ned_ts = pd.Timestamp(ned).normalize()
        except Exception:
            continue
        delta = (ned_ts - today).days
        if 0 <= delta <= window_days:
            blackout.add(row["ticker"])
    print(f"✅ استبعاد {len(blackout)} سهم بسبب الأرباح")
    return blackout


# ============================================================
# الفلاتر + المسح
# ============================================================
def apply_practical_filters(latest, meta, blackout_tickers, cfg, notes):
    keep = latest.copy()
    before = len(keep)
    keep = keep[keep["close"] > cfg.min_price]
    notes.append(f"Price > ${cfg.min_price}: {before} -> {len(keep)}")

    before = len(keep)
    keep = keep[keep["avg_vol_30d"] > cfg.min_avg_vol_30d]
    notes.append(f"Volume > {cfg.min_avg_vol_30d:,.0f}: {before} -> {len(keep)}")

    if meta is not None and "market_cap" in meta.columns:
        keep = keep.merge(meta[["ticker", "market_cap"]], on="ticker", how="left")
        before = len(keep)
        keep = keep[keep["market_cap"] > cfg.min_market_cap]
        notes.append(f"Market cap > ${cfg.min_market_cap:,.0f}: {before} -> {len(keep)}")

    if cfg.exclude_earnings_window and blackout_tickers:
        before = len(keep)
        keep = keep[~keep["ticker"].isin(blackout_tickers)]
        notes.append(f"Earnings blackout: {before} -> {len(keep)}")
    return keep


def jaccard(a, b):
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def run_screen_from_data(prices_df, meta_df, blackout_set, spy_df,
                          top_n=TOP_N, cfg=None,
                          min_qualified_score=MIN_SCORE,
                          jaccard_threshold=JACCARD_THRESHOLD):
    if cfg is None:
        cfg = FilterConfig()

    notes = []
    prices = prices_df.copy()

    regime, market_trend_up = detect_market_regime(spy_df)
    notes.append(f"Regime: {regime} | trend_up: {market_trend_up}")

    prices = prices.sort_values(["ticker", "date"])
    latest_rows = []
    feature_cache = {}
    for tkr, g in prices.groupby("ticker"):
        feat = compute_features(g)
        feature_cache[tkr] = feat
        last = feat.iloc[-1]
        latest_rows.append({
            "ticker": tkr,
            "close": last["close"],
            "avg_vol_30d": last["avg_vol_30d"] if not np.isnan(last["avg_vol_30d"]) else 0,
        })
    latest = pd.DataFrame(latest_rows)

    filtered = apply_practical_filters(latest, meta_df, blackout_set, cfg, notes)
    survivors = set(filtered["ticker"])

    results = []
    for tkr in survivors:
        feat = feature_cache[tkr]
        meta_row = None
        if meta_df is not None:
            m = meta_df[meta_df["ticker"] == tkr]
            if len(m):
                meta_row = m.iloc[0]
        r = score_ticker(feat, tkr, meta_row, market_trend_up, regime, DEFAULT_WEIGHTS)
        if r:
            results.append(r)

    clone_penalized = 0
    for i, r in enumerate(results):
        if not r.top_edges:
            continue
        clones = []
        for j, other in enumerate(results):
            if i == j or not other.top_edges:
                continue
            if jaccard(r.top_edges, other.top_edges) > jaccard_threshold:
                if other.score > r.score:
                    clones.append(other)
        if clones:
            penalty = max(0.75, 1 - 0.10 * len(clones))
            r.score = round(r.score * penalty, 1)
            clone_penalized += 1
    if clone_penalized:
        notes.append(f"Jaccard anti-clone: {clone_penalized} names penalized")

    fit_order = {"aligned": 0, "neutral": 1, "against": 2}
    results.sort(key=lambda r: (
        fit_order.get(r.regime_fit, 1),
        -r.score,
        -len(r.top_edges),
    ))

    qualified = [r for r in results
                 if r.score >= min_qualified_score and r.regime_fit == "aligned"]

    print(f"\nUniverse: {len(results)} | Qualified: {len(qualified)}")
    return qualified[:top_n], regime, notes


# ============================================================
# Google Sheets
# ============================================================
def get_gspread_client():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(SA_FILE, scopes=scope)
    return gspread.authorize(creds)


def get_or_create_ws(sh, title, rows=2000, cols=20):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=rows, cols=cols)


def _to_str(x):
    """تحويل أي قيمة إلى نص آمن لـ Google Sheets."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")
    if isinstance(x, timedelta):
        return str(x)
    return str(x)


def _df_to_rows(df):
    """تحويل DataFrame إلى list of lists (كل القيم نصية)."""
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(_to_str)
    return [df.columns.tolist()] + df.values.tolist()


def load_picks_from_sheets(client):
    """✅ A: تنظيف تلقائي للصفوف التالفة."""
    try:
        sh = client.open_by_key(SHEET_ID)
        ws = sh.worksheet("Picks History")
        data = ws.get_all_records()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)

        # ✅ A: إزالة الصفوف التالفة
        if "pick_date" in df.columns:
            s = df["pick_date"].astype(str).str.strip()
            df = df[(s != "") & (s != "NaT") & (s != "nan") & (s != "None")]
        if "ticker" in df.columns:
            df = df[df["ticker"].astype(str).str.strip() != ""]

        numeric_cols = ["score", "entry", "stop", "target",
                        "edge_count", "exit_price", "pnl_pct", "days_held"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ["pick_date", "exit_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        return df.reset_index(drop=True)
    except Exception as e:
        print(f"ℹ️ لا يوجد picks history بعد: {e}")
        return pd.DataFrame()


def save_picks_to_sheets(client, picks_df):
    sh = client.open_by_key(SHEET_ID)
    ws = get_or_create_ws(sh, "Picks History")
    ws.clear()
    if len(picks_df) == 0:
        return
    rows = _df_to_rows(picks_df)
    ws.update("A1", rows)


# ============================================================
# تتبع الإشارات
# ============================================================
def save_new_picks(results_list, regime, picks_history):
    today = pd.Timestamp.now().normalize()
    new_rows = []
    for r in results_list:
        style = ("Trend" if r.style_votes["trend"] > r.style_votes["mean_reversion"]
                 else "Mean-Reversion")
        new_rows.append({
            "pick_date": str(today.date()),
            "ticker": r.ticker,
            "score": r.score,
            "style": style,
            "regime_fit": r.regime_fit,
            "confidence": r.confidence,
            "top_edges": ", ".join(str(e) for e in r.top_edges),
            "edge_count": len(r.top_edges),
            "entry": r.entry,
            "stop": r.stop,
            "target": r.target,
            "regime": regime,
            "status": "open",
            "exit_date": "",
            "exit_price": "",
            "pnl_pct": "",
            "days_held": "",
            "reason": r.reason,
        })
    if not new_rows:
        return picks_history
    new_df = pd.DataFrame(new_rows)
    if len(picks_history) > 0:
        mask = ~((picks_history["pick_date"].astype(str) == str(today.date())) &
                 (picks_history["ticker"].isin(new_df["ticker"])))
        combined = pd.concat([picks_history[mask], new_df], ignore_index=True)
    else:
        combined = new_df
    return combined


def evaluate_open_picks(picks_history, prices_df, max_hold_days=HOLD_DAYS):
    """محصّنة: try/except لكل صف + logging."""
    if len(picks_history) == 0:
        print("ℹ️ picks_history فارغ — لا شيء لتقييمه")
        return picks_history

    if "status" not in picks_history.columns:
        print("⚠️ لا يوجد عمود status — تخطي التقييم")
        return picks_history

    prices_df = prices_df.copy()
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    evaluated = 0
    skipped = 0
    open_count = 0

    for idx, row in picks_history.iterrows():
        try:
            status_val = str(row.get("status", "")).strip()
            if status_val != "open":
                continue
            open_count += 1

            pick_date = pd.Timestamp(row["pick_date"])
            if pd.isna(pick_date):
                print(f"⚠️ تخطي صف {idx}: pick_date فارغ")
                skipped += 1
                continue

            ticker = str(row.get("ticker", "")).strip()
            if not ticker:
                skipped += 1
                continue

            try:
                entry = float(row["entry"])
                stop = float(row["stop"])
                target = float(row["target"])
            except (ValueError, TypeError) as e:
                print(f"⚠️ تخطي {ticker}: قيم غير صالحة ({e})")
                skipped += 1
                continue

            if pd.isna(entry) or pd.isna(stop) or pd.isna(target):
                print(f"⚠️ تخطي {ticker}: قيم NaN")
                skipped += 1
                continue

            sub = prices_df[(prices_df["ticker"] == ticker) &
                            (prices_df["date"] > pick_date)].sort_values("date")
            if len(sub) < max_hold_days:
                continue

            window = sub.head(max_hold_days).reset_index(drop=True)
            outcome = "timeout"
            exit_date = window.iloc[-1]["date"]
            exit_price = window.iloc[-1]["close"]

            for _, day in window.iterrows():
                hit_stop = day["low"] <= stop
                hit_target = day["high"] >= target
                if hit_stop and hit_target:
                    outcome, exit_price, exit_date = "loss", stop, day["date"]
                    break
                elif hit_stop:
                    outcome, exit_price, exit_date = "loss", stop, day["date"]
                    break
                elif hit_target:
                    outcome, exit_price, exit_date = "win", target, day["date"]
                    break

            pnl_pct = (exit_price - entry) / entry * 100
            days_held = (pd.Timestamp(exit_date) - pick_date).days

            picks_history.at[idx, "status"] = outcome
            picks_history.at[idx, "exit_date"] = str(pd.Timestamp(exit_date).date())
            picks_history.at[idx, "exit_price"] = round(float(exit_price), 2)
            picks_history.at[idx, "pnl_pct"] = round(float(pnl_pct), 2)
            picks_history.at[idx, "days_held"] = int(days_held)
            evaluated += 1

        except Exception as e:
            print(f"⚠️ خطأ في تقييم صف {idx}: {type(e).__name__}: {e}")
            skipped += 1
            continue

    print(f"✅ تقييم: {evaluated} محدّثة، {skipped} متخطاة، "
          f"{open_count} صف مفتوح")
    return picks_history


def compute_stats(picks_history):
    if len(picks_history) == 0:
        return None
    closed = picks_history[picks_history["status"].isin(["win", "loss", "timeout"])].copy()
    if len(closed) == 0:
        return None
    closed["pnl_pct"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    closed = closed.dropna(subset=["pnl_pct"])
    if len(closed) == 0:
        return None
    wins = (closed["status"] == "win").sum()
    return {
        "total": len(closed),
        "wins": int(wins),
        "losses": int((closed["status"] == "loss").sum()),
        "timeouts": int((closed["status"] == "timeout").sum()),
        "hit_rate": float(wins / len(closed) * 100),
        "expectancy": float(closed["pnl_pct"].mean()),
    }


# ============================================================
# ✅ B: الإيميل
# ============================================================
def send_email_report(results_list, stats, regime):
    """إرسال تقرير HTML بأفضل 3 أسهم + إحصاءات."""
    email_user = os.environ.get("EMAIL_USER")
    email_pass = os.environ.get("EMAIL_APP_PASSWORD")
    email_to = os.environ.get("EMAIL_TO")

    if not all([email_user, email_pass, email_to]):
        print("ℹ️ إعدادات الإيميل غير مكتملة — تم التخطي")
        return

    today_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    rows_html = ""
    for i, r in enumerate(results_list[:3], 1):
        style = ("Trend" if r.style_votes["trend"] > r.style_votes["mean_reversion"]
                 else "Mean-Reversion")
        edges = ", ".join(str(e) for e in r.top_edges)
        rows_html += f"""
        <tr>
          <td style="padding:8px;font-weight:bold">{i}. {r.ticker}</td>
          <td style="padding:8px">{r.score}</td>
          <td style="padding:8px">{style}</td>
          <td style="padding:8px">{r.confidence}</td>
          <td style="padding:8px;font-family:monospace">{r.entry} / {r.stop} / {r.target}</td>
          <td style="padding:8px;font-size:12px;color:#555">{edges}</td>
        </tr>"""

    if stats:
        stats_html = f"""
        <h3 style="margin-top:24px">📈 الأداء التراكمي</h3>
        <ul>
          <li>إجمالي الإشارات المغلقة: <b>{stats['total']}</b></li>
          <li>Wins: <b>{stats['wins']}</b> | Losses: <b>{stats['losses']}</b> | Timeouts: <b>{stats['timeouts']}</b></li>
          <li>Hit Rate: <b>{stats['hit_rate']:.1f}%</b></li>
          <li>Expectancy: <b>{stats['expectancy']:+.2f}%</b> لكل صفقة</li>
        </ul>"""
    else:
        stats_html = "<p style='color:#888'>لا توجد إشارات مغلقة بعد — التقييم يبدأ بعد 5 أيام تداول.</p>"

    body = f"""
    <html><body style="font-family:Arial,sans-serif;color:#222">
      <h2>📊 S&P 500 Multi-Edge Screen</h2>
      <p><b>التاريخ:</b> {today_str}<br>
         <b>النظام:</b> {regime}<br>
         <b>الإعدادات الصالحة:</b> {len(results_list)}</p>

      <h3>🏆 أفضل 3 أسهم</h3>
      <table style="border-collapse:collapse;border:1px solid #ddd;width:100%">
        <thead>
          <tr style="background:#f0f0f0">
            <th style="padding:8px">Ticker</th>
            <th style="padding:8px">Score</th>
            <th style="padding:8px">Style</th>
            <th style="padding:8px">Confidence</th>
            <th style="padding:8px">Entry/Stop/Target</th>
            <th style="padding:8px">Edges</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>

      {stats_html}

      <p style="margin-top:24px;font-size:12px;color:#888">
        🔗 الجدول الكامل:
        <a href="https://docs.google.com/spreadsheets/d/{SHEET_ID}">Google Sheets</a>
      </p>
    </body></html>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"📊 S&P Screen — {today_str[:10]}"
        msg["From"] = email_user
        msg["To"] = email_to
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_user, email_pass)
            server.send_message(msg)

        print(f"✅ تم إرسال الإيميل إلى {email_to}")
    except Exception as e:
        print(f"⚠️ فشل إرسال الإيميل: {e}")


# ============================================================
# ✅ C: الرسوم البيانية
# ============================================================
def create_performance_charts(client):
    """إنشاء رسم بياني في ورقة Performance."""
    try:
        sh = client.open_by_key(SHEET_ID)
        ws = sh.worksheet("Performance")
        sheet_id = ws._properties["sheetId"]

        all_values = ws.get_all_values()
        if len(all_values) < 6:
            print("ℹ️ بيانات غير كافية للرسم البياني")
            return

        # اقرأ الإحصاءات من الورقة
        stats_dict = {}
        for row in all_values:
            if len(row) >= 2:
                stats_dict[row[0]] = row[1]

        try:
            wins_val = int(float(stats_dict.get("Wins", 0) or 0))
            losses_val = int(float(stats_dict.get("Losses", 0) or 0))
            timeouts_val = int(float(stats_dict.get("Timeouts", 0) or 0))
        except Exception:
            wins_val = losses_val = timeouts_val = 0

        # اكتب البيانات المساعدة في D1:E4
        helper_data = [
            ["Status", "Count"],
            ["Wins", wins_val],
            ["Losses", losses_val],
            ["Timeouts", timeouts_val],
        ]
        ws.update("D1", helper_data)

        # احذف أي رسوم قديمة
        try:
            meta = sh.fetch_sheet_metadata()
            charts_to_delete = []
            for sheet in meta.get("sheets", []):
                if sheet["properties"]["sheetId"] == sheet_id:
                    for chart in sheet.get("charts", []):
                        charts_to_delete.append(
                            {"deleteEmbeddedObject": {"objectId": chart["chartId"]}}
                        )
            if charts_to_delete:
                sh.batch_update({"requests": charts_to_delete})
        except Exception:
            pass

        # أضف رسم دائري
        requests = [{
            "addChart": {
                "chart": {
                    "spec": {
                        "title": "توزيع نتائج الإشارات",
                        "pieChart": {
                            "legendPosition": "RIGHT_LEGEND",
                            "domain": {"sourceRange": {"sources": [{
                                "sheetId": sheet_id,
                                "startRowIndex": 1, "endRowIndex": 4,
                                "startColumnIndex": 3, "endColumnIndex": 4,
                            }]}},
                            "series": {"sourceRange": {"sources": [{
                                "sheetId": sheet_id,
                                "startRowIndex": 1, "endRowIndex": 4,
                                "startColumnIndex": 4, "endColumnIndex": 5,
                            }]}},
                        },
                    },
                    "position": {
                        "overlayPosition": {
                            "anchorCell": {
                                "sheetId": sheet_id,
                                "rowIndex": 12, "columnIndex": 3,
                            },
                            "widthPixels": 420, "heightPixels": 280,
                        }
                    },
                }
            }
        }]

        sh.batch_update({"requests": requests})
        print("✅ تم إنشاء الرسم البياني في Performance")
    except Exception as e:
        print(f"⚠️ فشل إنشاء الرسم: {e}")


# ============================================================
# التصدير
# ============================================================
def export_to_sheets(client, results_list, picks_history, stats, regime, notes):
    sh = client.open_by_key(SHEET_ID)
    today_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- Daily Screen ---
    ws = get_or_create_ws(sh, "Daily Screen")
    ws.clear()
    header = [
        [f"S&P 500 Multi-Edge Screen — {today_str}"],
        [f"Regime: {regime}"],
        [f"Qualified Setups: {len(results_list)}"],
        [],
    ]
    ws.update("A1", header)

    if results_list:
        rows = []
        for r in results_list:
            style = ("Trend" if r.style_votes["trend"] > r.style_votes["mean_reversion"]
                     else "Mean-Reversion")
            rows.append({
                "ticker": r.ticker,
                "score": r.score,
                "edge_count": len(r.top_edges),
                "style": style,
                "regime_fit": r.regime_fit,
                "confidence": r.confidence,
                "top_edges": ", ".join(str(e) for e in r.top_edges),
                "entry": r.entry,
                "stop": r.stop,
                "target": r.target,
                "reason": r.reason,
            })
        df = pd.DataFrame(rows)
        ws.update("A5", _df_to_rows(df))

    # --- Performance ---
    if stats:
        ws_p = get_or_create_ws(sh, "Performance")
        ws_p.clear()
        perf = [
            ["PERFORMANCE REPORT", ""],
            ["Generated at", today_str],
            [],
            ["Metric", "Value"],
            ["Total Closed", stats["total"]],
            ["Wins", stats["wins"]],
            ["Losses", stats["losses"]],
            ["Timeouts", stats["timeouts"]],
            ["Hit Rate (%)", round(stats["hit_rate"], 2)],
            ["Expectancy (% per trade)", round(stats["expectancy"], 3)],
        ]
        ws_p.update("A1", perf)

    # --- Open Positions ---
    if len(picks_history) > 0:
        open_picks = picks_history[picks_history["status"] == "open"]
        if len(open_picks) > 0:
            ws_o = get_or_create_ws(sh, "Open Positions")
            ws_o.clear()
            cols = ["ticker", "pick_date", "entry", "stop", "target",
                    "confidence", "style", "regime"]
            cols = [c for c in cols if c in open_picks.columns]
            ws_o.update("A1", [[f"Open Positions — {today_str}"]])
            sub = open_picks[cols].copy()
            ws_o.update("A3", _df_to_rows(sub))

    print(f"🔗 https://docs.google.com/spreadsheets/d/{SHEET_ID}")


# ============================================================
# main
# ============================================================
def main():
    print("=" * 70)
    print(f"S&P 500 Multi-Edge Screener — {datetime.now()}")
    print("=" * 70)

    if not SHEET_ID:
        print("❌ SHEET_ID غير موجود في المتغيرات")
        sys.exit(1)

    if not os.path.exists(SA_FILE):
        print(f"❌ ملف Service Account غير موجود: {SA_FILE}")
        sys.exit(1)

    client = get_gspread_client()
    print("✅ تم الاتصال بـ Google Sheets")

    tickers = get_sp500_tickers()
    if not tickers:
        print("❌ لا توجد قائمة أسهم")
        sys.exit(1)

    prices_wide = download_prices(tickers, period=PERIOD)
    if prices_wide.empty:
        print("❌ فشل تحميل الأسعار")
        sys.exit(1)

    long_df = prices_to_long(prices_wide)
    validate_latest_date(long_df)
    spy_df = download_spy(period=PERIOD)
    meta_df = fetch_metadata(tickers, max_tickers=MAX_META)
    blackout_set = build_earnings_blackout(meta_df, window_days=3)

    results_list, regime, notes = run_screen_from_data(
        prices_df=long_df,
        meta_df=meta_df,
        blackout_set=blackout_set,
        spy_df=spy_df,
        top_n=TOP_N,
    )

    print("\n--- Filter funnel ---")
    for n in notes:
        print(f"  - {n}")
    print(f"\n--- Top {len(results_list)} ---")
    for i, r in enumerate(results_list, 1):
        style = ("Trend" if r.style_votes["trend"] > r.style_votes["mean_reversion"]
                 else "Mean-Reversion")
        print(f"{i:>2}. {r.ticker:<6} Score: {r.score:>5.1f} "
              f"Style: {style:<15} Conf: {r.confidence:<7} "
              f"Edges: {len(r.top_edges)}")

    # ===== DIAGNOSTIC =====
    print("\n" + "=" * 70)
    print("🔍 DIAGNOSTIC — VLO status")
    print("=" * 70)
    vlo_data = long_df[long_df["ticker"] == "VLO"]
    if len(vlo_data) == 0:
        print("❌ VLO غير موجود في long_df!")
    else:
        last_vlo = vlo_data.sort_values("date").iloc[-1]
        print(f"VLO صفوف: {len(vlo_data)}")
        print(f"VLO آخر تاريخ: {last_vlo['date'].date()}")
        print(f"VLO آخر close: {last_vlo['close']}")
        print(f"VLO آخر volume: {last_vlo['volume']:,}")

    # ===== picks_history =====
    picks_history = load_picks_from_sheets(client)
    print(f"\n📥 picks_history: {len(picks_history)} صف")

    if len(picks_history) > 0:
        picks_history = evaluate_open_picks(picks_history, long_df)
        newly_closed = picks_history[picks_history["status"].isin(
            ["win", "loss", "timeout"])]
        print(f"✅ إشارات مغلقة إجمالًا: {len(newly_closed)}")

    picks_history = save_new_picks(results_list, regime, picks_history)
    save_picks_to_sheets(client, picks_history)
    print(f"💾 تم حفظ picks_history ({len(picks_history)} صف)")

    # ===== الإحصاءات والتصدير =====
    stats = compute_stats(picks_history)
    export_to_sheets(client, results_list, picks_history, stats, regime, notes)

    # ✅ C: إنشاء الرسوم
    create_performance_charts(client)

    # ✅ B: إرسال الإيميل
    send_email_report(results_list, stats, regime)

    print("\n✅ اكتمل التشغيل بنجاح")


if __name__ == "__main__":
    main()
