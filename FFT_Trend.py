import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import yfinance as yf
import sys

# ======== Configuration ========
SYMBOL = '^TNX'  # 10-Year Treasury Yield
START_DATE = '2023-01-01'
END_DATE = '2025-02-22'
PRICE_COLUMN = 'Close'
MIN_DATA_POINTS = 5
WINDOW_RATIO = 0.001
WINDOW_MIN = 10
THRESHOLD_RATIO = 0.00001

# ======== Visualization Setup ========
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)


# ======== Data Fetching ========
def fetch_financial_data():
    """Retrieve and validate financial data"""
    try:
        print(f"\nDownloading {SYMBOL} data ({START_DATE} to {END_DATE})...")
        df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, progress=False)

        if df.empty:
            raise ValueError("Empty dataset received. Please verify:")

        print("\nData successfully retrieved! First 5 rows:")
        print(df.head())

        if PRICE_COLUMN not in df.columns:
            raise ValueError(f"Column {PRICE_COLUMN} not found. Available columns: {df.columns.tolist()}")

        prices = df[PRICE_COLUMN].squeeze().dropna()

        if len(prices) < MIN_DATA_POINTS:
            raise ValueError(f"Insufficient data points ({len(prices)}). Minimum required: {MIN_DATA_POINTS}")

        return prices

    except Exception as e:
        print(f"\nData fetching failed: {str(e)}")
        sys.exit(1)


# ======== FFT Analysis ========
def perform_fft_analysis(prices):
    """Perform FFT-based frequency analysis using absolute changes"""
    try:
        # 使用绝对价格变动代替对数收益率
        price_changes = np.diff(prices.to_numpy())  # 修改点

        n = len(price_changes)
        fft_result = np.fft.fft(price_changes)
        freqs = np.fft.fftfreq(n)

        amplitudes = np.abs(fft_result)[:n // 2]
        frequencies = freqs[:n // 2]

        # Filter valid periods (5 days to 2 years)
        valid_mask = (1 / 730 < np.abs(frequencies)) & (np.abs(frequencies) < 1 / 5)
        filtered_amps = amplitudes.copy()
        filtered_amps[~valid_mask] = 0

        peaks, _ = find_peaks(filtered_amps, height=0.1 * np.max(filtered_amps))

        if len(peaks) == 0:
            print("Warning: No significant cycles detected. Using default 30-day period")
            return 30

        dominant_periods = 1 / np.abs(frequencies[peaks])
        return dominant_periods.mean()

    except Exception as e:
        print(f"FFT analysis failed: {str(e)}")
        sys.exit(1)


# ======== Trend Detection ========
def detect_trends(prices):
    """Adaptive trend detection with FFT"""
    try:
        dominant_period = perform_fft_analysis(prices)
        window_size = max(int(dominant_period * WINDOW_RATIO), WINDOW_MIN)
        threshold = THRESHOLD_RATIO * prices.std()

        print(f"\nAdaptive parameters:"
              f"\n- Dominant period: {dominant_period:.1f} days"
              f"\n- Detection window: {window_size} days"
              f"\n- Volatility threshold: {threshold:.4f}")

        values = prices.to_numpy()
        dates = prices.index.to_pydatetime()

        pivots = np.zeros(len(values), dtype=int)
        last_extreme = values[0]
        trend_direction = 0

        for i in range(window_size, len(values) - window_size):
            local_window = values[i - window_size:i + window_size]
            is_peak = values[i] == np.max(local_window)
            is_valley = values[i] == np.min(local_window)

            if is_peak or is_valley:
                price_change = abs(values[i] - last_extreme)
                if price_change > threshold:
                    if is_peak and trend_direction != 1:
                        pivots[i] = 1
                        last_extreme = values[i]
                        trend_direction = 1
                    elif is_valley and trend_direction != -1:
                        pivots[i] = -1
                        last_extreme = values[i]
                        trend_direction = -1

        idx = np.where(pivots != 0)[0]
        return dates[idx], values[idx], pivots[idx]

    except Exception as e:
        print(f"Trend detection failed: {str(e)}")
        sys.exit(1)


# ======== Visualization ========
def plot_analysis(prices, dates, levels, types):
    """生成综合分析图表"""
    # 创建3行2列的布局
    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])

    # 定义子图位置
    ax_price = fig.add_subplot(gs[0, :])  # 价格走势（占据第一行）
    ax_fft = fig.add_subplot(gs[1, 0])  # FFT频谱
    ax_duration = fig.add_subplot(gs[1, 1])  # 持续时间分布
    ax_change = fig.add_subplot(gs[2, 0])  # 价格变动分布（新增）
    ax_table = fig.add_subplot(gs[2, 1])  # 统计表格

    # ==== 价格走势图表 ====
    ax_price.plot(prices.index, prices.values, label='Price', lw=1)
    for i in range(len(dates) - 1):
        start_idx = np.where(prices.index == dates[i])[0][0]
        end_idx = np.where(prices.index == dates[i + 1])[0][0]
        segment = prices.iloc[start_idx:end_idx + 1]
        color = 'green' if types[i + 1] == 1 else 'red'
        ax_price.plot(segment.index, segment.values, c=color, lw=1.5)

    ax_price.scatter(dates[types == 1], levels[types == 1], c='lime', s=80,
                     edgecolors='black', label='Peaks')
    ax_price.scatter(dates[types == -1], levels[types == -1], c='red', s=80,
                     edgecolors='black', label='Valleys')
    ax_price.set_title('10Y UST Yield Trend Analysis using FFT')
    ax_price.legend()

    # ==== FFT频谱 ====
    price_changes = np.diff(prices)
    n = len(price_changes)
    freqs = np.fft.fftfreq(n)[:n // 2]
    amplitudes = np.abs(np.fft.fft(price_changes))[:n // 2]
    ax_fft.semilogx(1 / np.abs(freqs), amplitudes)
    ax_fft.set_xlabel('Period (Days)')
    ax_fft.set_ylabel('Amplitude')
    ax_fft.set_title('Absolute Change Spectrum')
    ax_fft.grid(True)

    # ==== 持续时间分布 ====
    durations = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    ax_duration.hist(durations, bins=100, color='teal', alpha=0.7)
    ax_duration.set_xlabel('Trend Duration (Days)')
    ax_duration.set_title('Duration Distribution')

    # ==== 新增价格变动分布 ====
    price_changes = np.abs(np.diff(levels))  # 使用趋势段的价格变动
    ax_change.hist(price_changes, bins=100, color='salmon', alpha=0.7)
    ax_change.set_xlabel('Price Change per Trend')
    ax_change.set_title('Price Change Distribution')

    # ==== 统计表格 ====
    stats_df = pd.DataFrame({
        'Total Trends': [len(durations)],
        'Avg Duration': [f"{np.mean(durations):.1f} days"],
        'Max Duration': [f"{np.max(durations)} days"],
        'Avg Change': [f"{np.mean(price_changes):.2f}"],
        'Max Change': [f"{np.max(price_changes):.2f}"]
    })
    ax_table.axis('off')
    table = ax_table.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * 5
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    plt.tight_layout()
    plt.show()

# ======== Main Execution ========
if __name__ == "__main__":
    try:
        prices = fetch_financial_data()
        dates, levels, types = detect_trends(prices)

        print(f"\nDetected {len(dates)} trend pivots")
        print("\nStatistical Summary:")
        durations = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        print(pd.DataFrame({
            'Duration': durations,
            'Price Change': np.abs(np.diff(levels)),
            'Trend Type': ['Up' if t == 1 else 'Down' for t in types[1:]]
        }).describe())

        plot_analysis(prices, dates, levels, types)

    except Exception as e:
        print(f"\nExecution failed: {str(e)}")
        sys.exit(1)