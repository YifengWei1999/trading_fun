import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import yfinance as yf
import torch
import torch.nn as nn
import sys
from scipy.stats import gaussian_kde

# ======== Configuration ========
SYMBOL = '^TNX'  # 10-Year Treasury Yield
START_DATE = '2022-01-01'
END_DATE = '2025-03-06'
PRICE_COLUMN = 'Close'
MIN_DATA_POINTS = 10  # Minimum data points required
BASE_WINDOW = 5  # Base observation window
THRESHOLD_RATIO = 0.01  # Trend confirmation sensitivity
VOLATILITY_FACTOR = 5  # Window adjustment sensitivity


# ======== Neural Network Architecture ========
class WindowOptimizer(nn.Module):
    """Self-contained adaptive window predictor"""

    def __init__(self, input_size=5):
        super().__init__()
        # LSTM layer: (batch, seq, features) -> (batch, seq, hidden)
        self.lstm = nn.LSTM(input_size, 32, batch_first=True)
        # Attention: (seq, batch, features) -> (seq, batch, features)
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4)
        # Final prediction layer
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # Input shape: (batch_size=1, seq_len=1, input_size=5)
        x, _ = self.lstm(x)  # Output: (1, 1, 32)
        x = x.transpose(0, 1)  # Convert to (seq_len=1, batch=1, features=32)
        x, _ = self.attention(x, x, x)  # Self-attention
        return torch.sigmoid(self.fc(x[-1]))  # Final prediction


def initialize_model():
    """Generate and save initial model"""
    model = WindowOptimizer()
    torch.save(model.state_dict(), 'window_model.pth')
    return model


# ======== Data Pipeline ========
def fetch_market_data():
    """Retrieve and validate market data"""
    try:
        print(f"\n🔍 Fetching {SYMBOL} data ({START_DATE} to {END_DATE})...")
        df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, progress=False)

        if df.empty:
            raise ValueError("Received empty dataset")

        print("\n✅ Data successfully retrieved:")
        print(f"   Period: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"   Records: {len(df)} trading days")

        prices = df[PRICE_COLUMN].squeeze().dropna()

        if len(prices) < MIN_DATA_POINTS:
            raise ValueError(f"Insufficient data ({len(prices)}/{MIN_DATA_POINTS})")

        return prices

    except Exception as e:
        print(f"\n❌ Data Error: {str(e)}")
        sys.exit(1)


# ======== Core Analytics ========
def calculate_adaptive_window(prices, model):
    """Dynamic window size calculation"""
    with torch.no_grad():
        # Create 3D input tensor [batch=1, sequence=1, features=5]
        features = torch.tensor([[
            prices.pct_change().std() * np.sqrt(252),  # Annualized volatility
            (prices.diff() > 0).mean(),  # Upward probability
            len(find_peaks(prices.values)[0]),  # Local extrema count
            prices.autocorr(lag=5),  # Short-term correlation
            prices.rolling(20).std().iloc[-1]  # Recent volatility
        ]], dtype=torch.float32).unsqueeze(1)  # Add sequence dimension

        # Validate input dimensions
        if features.shape != (1, 1, 5):
            raise ValueError(f"Invalid feature shape {features.shape}, expected (1,1,5)")

        # Predict window adjustment factor
        adjustment = model(features).item()

    # Calculate and clamp window size
    window_size = int(BASE_WINDOW * (1 + adjustment))
    return max(min(window_size, 60), 5)  # Keep between 5-60 days


def spectral_analysis(prices):
    """FFT-based market cycle detection"""
    try:
        changes = np.diff(prices.to_numpy())
        n = len(changes)

        fft_result = np.fft.fft(changes)
        freqs = np.fft.fftfreq(n)

        # Filter meaningful cycles (15 days to 2 years)
        valid_mask = (1 / 730 < np.abs(freqs)) & (np.abs(freqs) < 1 / 15)
        if not np.any(valid_mask):
            return BASE_WINDOW

        dominant_freq = freqs[valid_mask][np.argmax(np.abs(fft_result)[valid_mask])]
        return int(1 / np.abs(dominant_freq))

    except Exception as e:
        print(f"⚠️ Spectral analysis warning: {str(e)}")
        return BASE_WINDOW


# ======== Trend Detection Engine ========
def detect_market_trends(prices, model):
    """Adaptive trend detection system"""
    try:
        # Calculate dynamic parameters
        window_size = calculate_adaptive_window(prices, model)
        threshold = THRESHOLD_RATIO * prices.std()

        print(f"\n⚙️ Adaptive Parameters:")
        print(f"   Detection Window: {window_size} days")
        print(f"   Volatility Threshold: {threshold:.4f}")

        values = prices.to_numpy()
        dates = prices.index.to_pydatetime()
        pivots = np.zeros(len(values), dtype=int)
        last_extreme = values[0]
        trend_direction = 0  # 0-initial, 1-up, -1-down

        # Main detection loop
        for i in range(window_size, len(values) - window_size):
            # Local window analysis
            local_window = values[i - window_size:i + window_size]
            current_value = values[i]

            # Peak/Valley detection
            is_peak = (current_value == np.max(local_window)) and (trend_direction != 1)
            is_valley = (current_value == np.min(local_window)) and (trend_direction != -1)

            # Threshold validation
            if (is_peak or is_valley) and (abs(current_value - last_extreme) > threshold):
                pivots[i] = 1 if is_peak else -1
                last_extreme = current_value
                trend_direction = 1 if is_peak else -1

        # Extract valid pivots
        idx = np.where(pivots != 0)[0]
        return dates[idx], values[idx], pivots[idx]

    except Exception as e:
        print(f"\n❌ Trend Detection Error: {str(e)}")
        sys.exit(1)


# ======== Visualization System ========
def render_dashboard(prices, dates, levels, types):
    """Interactive Analysis Dashboard (with KDE distributions)"""
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])

    # ===== Price Chart =====
    ax_price = fig.add_subplot(gs[0, :])
    ax_price.plot(prices.index, prices, label='Price', alpha=0.3)
    for i in range(len(dates) - 1):
        segment = prices.loc[dates[i]:dates[i + 1]]
        color = 'green' if types[i + 1] == 1 else 'red'
        ax_price.plot(segment.index, segment, color=color, lw=2)
    ax_price.scatter(dates, levels, c=np.where(types == 1, 'lime', 'red'),
                     s=80, edgecolors='black', zorder=5)
    ax_price.set_title('10y UST Yield Adaptive Trend Analysis', fontsize=14)

    # ===== FFT Spectrum =====
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
    ax_fft.set_title('Market Rhythm Spectrum Analysis')
    ax_fft.grid(True)

    # ===== Duration Distribution =====
    ax_duration = fig.add_subplot(gs[1, 1])
    durations = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]

    # KDE calculation
    kde_duration = gaussian_kde(durations)
    xmin, xmax = min(durations), max(durations)
    x = np.linspace(xmin, xmax, 500)

    # Dual visualization
    ax_duration.hist(durations, bins=100, density=True, color='teal', alpha=0.3)
    ax_duration.plot(x, kde_duration(x), color='darkblue', lw=2, label='KDE Estimate')
    ax_duration.set_xlabel('Trend Duration (Days)')
    ax_duration.set_title('Duration Probability Density Distribution')
    ax_duration.legend()

    # ===== Price Change Distribution =====
    ax_change = fig.add_subplot(gs[2, :])
    changes = np.abs(np.diff(levels))

    # KDE with Silverman's rule
    if len(changes) > 1:
        kde_change = gaussian_kde(changes, bw_method='silverman')
        xc = np.linspace(0, max(changes) * 1.1, 500)
        ax_change.plot(xc, kde_change(xc), color='darkred', lw=2, label='KDE Estimate')
    else:  # Handle edge case
        ax_change.text(0.5, 0.5, 'Insufficient data for distribution', ha='center')

    # Combined visualization
    ax_change.hist(changes, bins=100, density=True, color='salmon', alpha=0.3)
    ax_change.set_xlabel('Price Change Magnitude')
    ax_change.set_title('Price Change Probability Density Distribution')
    ax_change.legend()

    plt.tight_layout()
    plt.show()


# ======== Main Execution Flow ========
if __name__ == "__main__":
    try:
        # Initialize self-contained model
        print("\n🔧 Initializing adaptive model...")
        model = initialize_model()

        # Data pipeline
        prices = fetch_market_data()

        # Trend detection
        dates, levels, types = detect_market_trends(prices, model)

        # Result validation
        if len(dates) < 2:
            print("\n⚠️ No significant trends detected")
            print("Potential reasons:")
            print("- Current market volatility is too low")
            print("- Analysis period is too short")
            print("- Try adjusting THRESHOLD_RATIO parameter")
            sys.exit(0)

        # Generate report
        print("\n📊 Trend Analysis Report:")
        print(f" Identified Trends: {len(dates) - 1}")
        avg_duration = np.mean([(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)])
        print(f" Average Duration: {avg_duration:.1f} days")
        print(f" Maximum Price Change: {np.max(np.abs(np.diff(levels))):.2f}")

        # Visualization
        print("\n🎨 Generating interactive dashboard...")
        render_dashboard(prices, dates, levels, types)

    except Exception as e:
        print(f"\n💥 Critical Error: {str(e)}")
        sys.exit(1)