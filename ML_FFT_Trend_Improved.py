import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import yfinance as yf
import torch
import torch.nn as nn
import sys

# ======== Configuration ========
SYMBOL = '^TNX'  # 10-Year Treasury Yield
START_DATE = '2023-01-01'
END_DATE = '2025-03-14'
PRICE_COLUMN = 'Close'
OHLC_COLUMNS = ['Open', 'High', 'Low', 'Close']
MIN_DATA_POINTS = 60
BASE_WINDOW = 30
THRESHOLD_RATIO = 0.18
VOLATILITY_FACTOR = 8


# ======== Neural Network Architecture ========
class WindowOptimizer(nn.Module):
    """Self-learning window size predictor with attention mechanism"""

    def __init__(self, input_size=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 32, batch_first=True)
        self.attention = nn.MultiheadAttention(32, 4)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.transpose(0, 1)
        x, _ = self.attention(x, x, x)
        return torch.sigmoid(self.fc(x[-1]))


def initialize_model():
    """Create and save initial model weights"""
    model = WindowOptimizer()
    torch.save(model.state_dict(), 'window_model.pth')
    return model


# ======== Data Pipeline ========
def fetch_market_data():
    """Retrieve and validate OHLC market data"""
    try:
        print(f"\n🔍 Fetching {SYMBOL} data ({START_DATE} to {END_DATE})...")
        df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, progress=False)

        if df.empty:
            raise ValueError("Empty dataset received from source")

        print("\n✅ Data successfully retrieved:")
        print(f"   Period: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"   Records: {len(df)} trading days")

        if not all(col in df.columns for col in OHLC_COLUMNS):
            missing = [col for col in OHLC_COLUMNS if col not in df.columns]
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        return df

    except Exception as e:
        print(f"\n❌ Data Error: {str(e)}")
        sys.exit(1)


# ======== Core Analytics ========
def calculate_atr(df, window=14):
    """Calculate Average True Range"""
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift()), abs(low - close.shift()))
    )
    return tr.rolling(window).mean()


def calculate_adx(df, window=14):
    """Calculate Average Directional Index"""
    high = df['High']
    low = df['Low']
    close = df['Close']

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr = calculate_atr(df, window)
    plus_di = 100 * (plus_dm / atr).ewm(span=window).mean()
    minus_di = 100 * (minus_dm / atr).ewm(span=window).mean()
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.ewm(span=window).mean()


def calculate_adaptive_window(prices, model, df):
    """Dynamic window calculation with volatility scaling"""
    with torch.no_grad():
        features = torch.tensor([[
            prices.diff().std() * np.sqrt(252),
            (prices.diff() > 0).mean(),
            len(find_peaks(prices.values)[0]),
            prices.autocorr(lag=5),
        ]], dtype=torch.float32).unsqueeze(1)

        if features.shape != (1, 1, 4):
            raise ValueError(f"Invalid feature shape {features.shape}")

        adjustment = model(features).item()

    calculated_window = int(BASE_WINDOW * (1 + adjustment))
    return max(min(calculated_window, 60), 5)


# ======== Enhanced Trend Detection ========
def detect_market_trends(df, model):
    """Multi-stage trend detection engine"""
    try:
        prices = df[PRICE_COLUMN].squeeze()
        main_window = calculate_adaptive_window(prices, model, df)
        confirmation_window = max(int(main_window * 0.3), 5)
        atr = calculate_atr(df).iloc[-1]
        threshold = THRESHOLD_RATIO * atr
        adx = calculate_adx(df).iloc[-1]

        print(f"\n⚙️ Adaptive Parameters:")
        print(f"   Main Window: {main_window} days")
        print(f"   Confirmation Window: {confirmation_window} days")
        print(f"   Volatility Threshold: {threshold[SYMBOL]:.4f}")
        print(f"   Trend Strength (ADX): {adx[SYMBOL]:.1f}")

        values = prices.to_numpy()
        dates = df.index.to_pydatetime()
        pivots = np.zeros(len(values), dtype=int)
        last_extreme = values[0]
        trend_direction = 0

        # Trend strength filter
        if adx < 25:
            print("⚠️ Weak trending market - applying conservative filters")
            threshold *= 1.5
            confirmation_window = int(confirmation_window * 1.2)

        for i in range(main_window, len(values) - main_window):
            # Multi-window validation
            main_peak = (values[i] == np.max(values[i - main_window:i + main_window]))
            main_valley = (values[i] == np.min(values[i - main_window:i + main_window]))
            short_peak = (values[i] == np.max(values[i - confirmation_window:i + confirmation_window]))
            short_valley = (values[i] == np.min(values[i - confirmation_window:i + confirmation_window]))

            # Threshold validation
            price_change = abs(values[i] - last_extreme)
            pullback = price_change / last_extreme if last_extreme != 0 else 0

            if (main_peak or main_valley) and (price_change > threshold) and (pullback > 0.015):
                # Time-series confirmation
                confirm_window = values[i:i + 3]
                valid_peak = (main_peak and short_peak and
                              all(confirm_window < values[i]))
                valid_valley = (main_valley and short_valley and
                                all(confirm_window > values[i]))

                # State management
                if valid_peak and trend_direction != 1:
                    pivots[i] = 1
                    last_extreme = values[i]
                    trend_direction = 1
                elif valid_valley and trend_direction != -1:
                    pivots[i] = -1
                    last_extreme = values[i]
                    trend_direction = -1

        idx = np.where(pivots != 0)[0]
        return dates[idx], values[idx], pivots[idx]

    except Exception as e:
        print(f"\n❌ Trend Detection Error: {str(e)}")
        sys.exit(1)


# ======== Advanced Visualization ========
def render_dashboard(df, dates, levels, types):
    """Interactive analytical dashboard"""
    prices = df[PRICE_COLUMN]
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])

    # Price trajectory
    ax_price = fig.add_subplot(gs[0, :])
    ax_price.plot(prices.index, prices, label='Price', alpha=0.3)
    for i in range(len(dates) - 1):
        segment = prices.loc[dates[i]:dates[i + 1]]
        color = 'green' if types[i + 1] == 1 else 'red'
        ax_price.plot(segment.index, segment, color=color, lw=2)
    ax_price.scatter(dates, levels, c=np.where(types == 1, 'lime', 'red'),
                     s=80, edgecolors='black', zorder=5)
    ax_price.set_title('Adaptive Trend Analysis', fontsize=14)

    # FFT spectrum
    ax_fft = fig.add_subplot(gs[1, 0])
    price_changes = np.diff(prices)
    n = len(price_changes)
    freqs = np.fft.fftfreq(n)[:n // 2]
    amplitudes = np.abs(np.fft.fft(price_changes))[:n // 2]
    periods = 1 / (np.abs(freqs) + 1e-10)
    valid = (periods > 5) & (periods < 730)
    ax_fft.semilogx(periods[valid], amplitudes[valid])
    ax_fft.set_xlabel('Cycle Length (Days)')
    ax_fft.set_ylabel('Amplitude')
    ax_fft.set_title('Market Rhythm Spectrum')
    ax_fft.grid(True)

    # Duration distribution
    ax_duration = fig.add_subplot(gs[1, 1])
    durations = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    kde_duration = gaussian_kde(durations)
    x = np.linspace(min(durations), max(durations), 500)
    ax_duration.hist(durations, bins=15, density=True, color='teal', alpha=0.3)
    ax_duration.plot(x, kde_duration(x), color='darkblue', lw=2, label='KDE')
    ax_duration.set_xlabel('Trend Duration (Days)')
    ax_duration.set_title('Duration Probability Distribution')
    ax_duration.legend()

    # Price change distribution
    ax_change = fig.add_subplot(gs[2, :])
    changes = np.abs(np.diff(levels))
    if len(changes) > 1:
        kde_change = gaussian_kde(changes, bw_method='silverman')
        x = np.linspace(0, max(changes) * 1.1, 500)
        ax_change.plot(x, kde_change(x), color='darkred', lw=2, label='KDE')
    ax_change.hist(changes, bins=15, density=True, color='salmon', alpha=0.3)
    ax_change.set_xlabel('Price Change Magnitude')
    ax_change.set_title('Price Movement Probability Distribution')
    ax_change.legend()

    plt.tight_layout()
    plt.show()


# ======== Execution Flow ========
if __name__ == "__main__":
    try:
        # Initialize components
        model = initialize_model()
        df = fetch_market_data()

        # Core analysis
        dates, levels, types = detect_market_trends(df, model)

        # Result validation
        if len(dates) < 2:
            print("\n⚠️ No significant trends detected")
            print("Potential reasons:")
            print("- Low market volatility")
            print("- Strong mean reversion pattern")
            sys.exit(0)

        # Generate report
        print("\n📊 Trend Analysis Report:")
        print(f" Identified Trends: {len(dates) - 1}")
        durations = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        print(f" Average Duration: {np.mean(durations):.1f} days")
        print(f" Maximum Price Change: {np.max(np.abs(np.diff(levels))):.2f}")

        # Visualization
        print("\n🎨 Generating analytical dashboard...")
        render_dashboard(df, dates, levels, types)

    except Exception as e:
        print(f"\n💥 Critical Error: {str(e)}")
        sys.exit(1)