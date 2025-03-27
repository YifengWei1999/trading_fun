import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pywt
import pymc as pm
import arviz as az
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from datetime import datetime, timedelta

# ======== Configuration ========
SYMBOL = '^TNX'  # 10-Year Treasury Yield
START_DATE = '2020-01-01'  # Extended history for better wavelet analysis
END_DATE = datetime.now().strftime('%Y-%m-%d')
PRICE_COLUMN = 'Close'
WAVELET_TYPE = 'cmor1.5-1.0'  # Complex Morlet wavelet
MAX_SCALE = 256  # Maximum scale for wavelet transform
MCMC_SAMPLES = None  # Remove MCMC configuration since we're using VI


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
        return prices

    except Exception as e:
        print(f"\n❌ Data Error: {str(e)}")
        return None


# ======== Wavelet Analysis ========
def perform_wavelet_analysis(prices):
    """Perform continuous wavelet transform on price data"""
    # Normalize data for better wavelet analysis
    normalized_data = (prices - prices.mean()) / prices.std()

    # Generate scales for wavelet transform (logarithmic spacing)
    scales = np.logspace(0, np.log10(MAX_SCALE), num=128)

    # Perform continuous wavelet transform
    coefficients, frequencies = pywt.cwt(normalized_data.values, scales, WAVELET_TYPE)

    # Convert frequencies to periods (in trading days)
    periods = 1 / frequencies

    return coefficients, periods, normalized_data


def identify_dominant_cycles(coefficients, periods):
    """Identify dominant cycles from wavelet power spectrum"""
    # Calculate wavelet power (squared absolute value)
    power = np.abs(coefficients) ** 2

    # Average power across time
    mean_power = power.mean(axis=1)

    # Find peaks in the power spectrum
    peak_indices = np.where((mean_power[1:-1] > mean_power[:-2]) &
                            (mean_power[1:-1] > mean_power[2:]))[0] + 1

    # Extract dominant periods
    dominant_periods = periods[peak_indices]
    dominant_powers = mean_power[peak_indices]

    # Sort by power
    sorted_indices = np.argsort(dominant_powers)[::-1]
    top_periods = dominant_periods[sorted_indices][:5]  # Top 5 dominant cycles

    return top_periods, mean_power


# ======== Bayesian Trend Analysis ========
def build_bayesian_model(trends):
    """Build Bayesian model using Variational Inference instead of MCMC"""
    durations = np.array([trend['duration'] for trend in trends])
    magnitudes = np.array([trend['magnitude'] for trend in trends])
    directions = np.array([trend['direction'] for trend in trends])

    with pm.Model() as model:
        # Priors for uptrend durations
        up_mu = pm.Gamma('up_duration_mu', alpha=2, beta=0.1)
        up_sigma = pm.HalfNormal('up_duration_sigma', sigma=10)

        # Priors for downtrend durations
        down_mu = pm.Gamma('down_duration_mu', alpha=2, beta=0.1)
        down_sigma = pm.HalfNormal('down_duration_sigma', sigma=10)

        # Priors for magnitude
        mag_mu = pm.Normal('magnitude_mu', mu=0, sigma=1)
        mag_sigma = pm.HalfNormal('magnitude_sigma', sigma=1)

        # Likelihood for durations based on trend direction
        up_duration = pm.Gamma('up_duration',
                               alpha=up_mu ** 2 / up_sigma ** 2,
                               beta=up_mu / up_sigma ** 2,
                               observed=durations[directions == 1])

        down_duration = pm.Gamma('down_duration',
                                 alpha=down_mu ** 2 / down_sigma ** 2,
                                 beta=down_mu / down_sigma ** 2,
                                 observed=durations[directions == -1])

        # Likelihood for magnitudes
        magnitude = pm.Normal('magnitude', mu=mag_mu, sigma=mag_sigma,
                              observed=magnitudes)

        # Use Variational Inference instead of MCMC
        approx = pm.fit(n=10000, method='advi')  # Faster than MCMC
        trace = approx.sample(1000)  # Sample from the approximated posterior

    return trace


def extract_trends(prices, min_trend_duration=3):
    """
    Enhanced trend extraction using adaptive thresholds and peak detection
    """

    def merge_close_points(points, min_distance=2):
        """Merge pivot points that are too close together"""
        if not points:
            return points

        points = sorted(points)  # Ensure points are sorted
        merged = [points[0]]

        for point in points[1:]:
            if (point - merged[-1]).days >= min_distance:
                merged.append(point)
        return merged

    def calculate_adaptive_threshold(data, window=5):
        """Calculate adaptive threshold with more weight on recent data"""
        # Calculate rolling standard deviation
        rolling_std = data.rolling(window=window).std()

        # Calculate exponentially weighted standard deviation (more weight on recent data)
        exp_weighted_std = data.ewm(span=20).std()

        # Combine both metrics with more weight on recent data
        recent_threshold = rolling_std.iloc[-60:].mean() * 0.8  # Last 3 months
        full_threshold = rolling_std.mean() * 0.8

        # Use 70% weight on recent threshold
        return 0.7 * recent_threshold + 0.3 * full_threshold

    def identify_pivot_points(prices, threshold):
        """Identify potential trend pivot points using price movements"""
        highs = []
        lows = []

        # Calculate smoothed prices with less smoothing for recent data
        smoothed = pd.Series(
            gaussian_filter1d(prices.values, sigma=1),
            index=prices.index
        )

        # Special handling for recent data (last 60 days)
        recent_threshold = threshold * 0.7  # More sensitive threshold for recent data
        last_60_days = prices.index[-60:]

        for i in range(2, len(smoothed) - 2):
            p = smoothed.iloc[i]
            prev2, prev1 = smoothed.iloc[i - 2], smoothed.iloc[i - 1]
            next1, next2 = smoothed.iloc[i + 1], smoothed.iloc[i + 2]

            # Use more sensitive threshold for recent data
            current_threshold = recent_threshold if smoothed.index[i] in last_60_days else threshold

            if prev2 < prev1 < p > next1 > next2 and abs(p - prev2) > current_threshold:
                highs.append(smoothed.index[i])
            elif prev2 > prev1 > p < next1 < next2 and abs(p - prev2) > current_threshold:
                lows.append(smoothed.index[i])

        # Always check the last point
        last_idx = len(smoothed) - 1
        if last_idx >= 4:
            last_points = smoothed.iloc[-5:]
            if last_points.iloc[-1] == last_points.max():
                highs.append(smoothed.index[-1])
            elif last_points.iloc[-1] == last_points.min():
                lows.append(smoothed.index[-1])

        return highs, lows

    def validate_trend(trend, prices):
        """Validate trend data with special handling for recent trends"""
        start_date = trend['start']
        end_date = trend['end']
        segment = prices.loc[start_date:end_date]

        # Calculate trend metrics
        start_price = segment.iloc[0]
        end_price = segment.iloc[-1]
        price_change = end_price - start_price
        direction = 1 if price_change > 0 else -1

        # Calculate trend strength
        x = np.arange(len(segment))
        y = segment.values
        slope, _, r_value, _, _ = stats.linregress(x, y)
        strength = abs(r_value)

        # More lenient validation for recent trends (last 60 days)
        is_recent = (end_date >= prices.index[-60])
        min_strength = 0.3 if is_recent else 0.4
        min_price_change = prices.std() * (0.08 if is_recent else 0.1)

        # Validate the trend
        if abs(price_change) < min_price_change or strength < min_strength:
            return None

        return {
            'start': start_date,
            'end': end_date,
            'duration': (end_date - start_date).days,
            'magnitude': abs(price_change),
            'direction': direction,
            'strength': strength,
            'slope': slope,
            'start_price': start_price,
            'end_price': end_price,
            'is_recent': is_recent
        }

    # Main trend extraction process
    threshold = calculate_adaptive_threshold(prices)
    highs, lows = identify_pivot_points(prices, threshold)

    print(f"\nInitial pivot points found - Highs: {len(highs)}, Lows: {len(lows)}")

    # Ensure the last point is included if it's significant
    last_date = prices.index[-1]
    if last_date not in highs and last_date not in lows:
        # Compare with recent history
        last_week = prices.last('5D')
        if prices.iloc[-1] == last_week.max():
            highs.append(last_date)
        elif prices.iloc[-1] == last_week.min():
            lows.append(last_date)

    highs = merge_close_points(highs)
    lows = merge_close_points(lows)

    print(f"After merging - Highs: {len(highs)}, Lows: {len(lows)}")

    all_pivots = sorted(set(highs + lows))

    # Ensure we have the latest date
    if all_pivots and all_pivots[-1] != last_date:
        all_pivots.append(last_date)

    # Extract and validate trends
    trends = []
    for i in range(len(all_pivots) - 1):
        start_date = all_pivots[i]
        end_date = all_pivots[i + 1]
        segment = prices.loc[start_date:end_date]

        if len(segment) >= min_trend_duration:
            trend = validate_trend({
                'start': start_date,
                'end': end_date
            }, prices)

            if trend is not None:
                trends.append(trend)

    # Add debug information for the last trend
    if trends:
        last_trend = trends[-1]
        print("\nLast identified trend:")
        print(f"Start: {last_trend['start'].date()}")
        print(f"End: {last_trend['end'].date()}")
        print(f"Duration: {last_trend['duration']} days")
        print(f"Direction: {'Up' if last_trend['direction'] == 1 else 'Down'}")
        print(f"Strength: {last_trend['strength']:.2f}")
        print(f"Price change: {last_trend['end_price'] - last_trend['start_price']:.4f}")

    return trends


def calculate_trend_probabilities(trace, current_trend):
    """Calculate probability of trend continuation using VI results"""
    # Extract posterior distributions (same as before)
    if current_trend['direction'] == 1:
        duration_samples = trace.posterior['up_duration_mu'].values.flatten()
    else:
        duration_samples = trace.posterior['down_duration_mu'].values.flatten()

    magnitude_mu = trace.posterior['magnitude_mu'].values.flatten()
    magnitude_sigma = trace.posterior['magnitude_sigma'].values.flatten()

    # Current trend duration
    current_duration = current_trend['duration']

    # Calculate probability of trend continuing for different time periods
    continuation_probs = {}
    for days in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]:
        # Use mean and std of the VI approximation for faster computation
        mu = np.mean(duration_samples)
        sigma = np.std(duration_samples)
        z_score = (current_duration + days - mu) / (sigma + 1e-6)
        prob = 1 - stats.norm.cdf(z_score)
        continuation_probs[days] = max(0.1, min(0.9, prob))  # Bound probabilities

    # Calculate expected magnitude change
    expected_magnitude = np.mean(magnitude_mu)
    magnitude_uncertainty = np.mean(magnitude_sigma)

    return continuation_probs, expected_magnitude, magnitude_uncertainty


# ======== Visualization ========
def plot_wavelet_analysis(coefficients, periods, prices):
    """Plot wavelet transform results"""
    plt.figure(figsize=(15, 10))

    # Plot 1: Original price data
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(prices.index, prices, 'k-')
    ax1.set_title('10-Year Treasury Yield')
    ax1.set_ylabel('Yield (%)')

    # Plot 2: Wavelet power spectrum
    ax2 = plt.subplot(3, 1, 2)
    power = np.abs(coefficients) ** 2
    levels = np.linspace(0, power.max(), 100)

    # Convert dates to numerical values for plotting
    t = np.arange(len(prices))
    T, S = np.meshgrid(t, periods)

    cs = ax2.contourf(T, S, power, levels=levels, cmap='viridis')
    ax2.set_ylabel('Period (days)')
    ax2.set_title('Wavelet Power Spectrum')
    ax2.set_yscale('log')
    ax2.set_ylim([min(periods), max(periods)])

    # Plot 3: Global wavelet spectrum
    ax3 = plt.subplot(3, 1, 3)
    mean_power = power.mean(axis=1)
    ax3.plot(mean_power, periods, 'k-')
    ax3.set_xlabel('Power')
    ax3.set_ylabel('Period (days)')
    ax3.set_title('Global Wavelet Spectrum')
    ax3.set_yscale('log')
    ax3.set_ylim([min(periods), max(periods)])

    plt.tight_layout()


def plot_bayesian_results(trace, trends, current_trend, continuation_probs):
    """Plot Bayesian analysis results"""
    # Plot 1: Posterior distributions
    az.plot_posterior(trace, var_names=['up_duration_mu', 'down_duration_mu', 'magnitude_mu'])
    plt.tight_layout()
    plt.show()

    # Plot 2: Trend continuation probability
    plt.figure(figsize=(10, 6))
    days = list(continuation_probs.keys())
    probs = list(continuation_probs.values())

    plt.bar(days, probs, color='skyblue')
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Additional Days')
    plt.ylabel('Continuation Probability')
    plt.title(f"Probability of {'Uptrend' if current_trend['direction'] == 1 else 'Downtrend'} Continuing")
    plt.xticks(days)
    plt.ylim(0, 1)

    for i, prob in enumerate(probs):
        plt.text(days[i], prob + 0.02, f'{prob:.2f}', ha='center')

    plt.tight_layout()


def plot_trends_on_price(prices, trends):
    """Enhanced plot with trend information"""
    plt.figure(figsize=(15, 8))

    # Plot price data
    plt.plot(prices.index, prices, 'k-', alpha=0.5, label='Price', linewidth=1)

    # Debug information in console
    print(f"\nFound {len(trends)} trends to plot")

    # Plot trends with enhanced visualization
    for i, trend in enumerate(trends):
        start_date = trend['start']
        end_date = trend['end']
        trend_data = prices.loc[start_date:end_date]

        # Debug information in console
        print(f"\nTrend {i + 1}:")
        print(f"Start: {start_date.date()}")
        print(f"End: {end_date.date()}")
        print(f"Duration: {trend['duration']} days")
        print(f"Direction: {'Up' if trend['direction'] == 1 else 'Down'}")
        print(f"Strength: {trend['strength']:.2f}")

        # Color based on trend direction and strength
        alpha = min(1.0, trend['strength'])
        color = 'green' if trend['direction'] == 1 else 'red'

        # Plot trend line with increased visibility
        plt.plot(trend_data.index, trend_data, color=color,
                 alpha=max(0.7, alpha), linewidth=2.5)

    plt.title('Price with Identified Trends', fontsize=12, pad=20)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()


def plot_trend_probabilities(continuation_probs, expected_magnitude, magnitude_uncertainty):
    """Plot trend continuation probabilities and magnitude expectations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Trend Continuation Probability
    days = list(continuation_probs.keys())
    probs = list(continuation_probs.values())

    ax1.plot(days, probs, 'b-', linewidth=2, marker='o')
    ax1.fill_between(days,
                     [max(0, p - 0.1) for p in probs],
                     [min(1, p + 0.1) for p in probs],
                     alpha=0.2, color='blue')

    ax1.set_xlabel('Days Ahead', fontsize=10)
    ax1.set_ylabel('Continuation Probability', fontsize=10)
    ax1.set_title('Trend Continuation Probability vs Time', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Add probability values
    for i, prob in enumerate(probs):
        ax1.annotate(f'{prob:.2f}',
                     (days[i], probs[i]),
                     xytext=(0, 5),
                     textcoords='offset points',
                     ha='center',
                     fontsize=8)

    # Plot 2: Expected Magnitude Distribution
    x = np.linspace(expected_magnitude - 3 * magnitude_uncertainty,
                    expected_magnitude + 3 * magnitude_uncertainty,
                    100)
    y = stats.norm.pdf(x, expected_magnitude, magnitude_uncertainty)

    ax2.plot(x, y, 'r-', linewidth=2)
    ax2.fill_between(x, y, alpha=0.2, color='red')

    ax2.axvline(expected_magnitude, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(expected_magnitude - magnitude_uncertainty, color='r', linestyle=':', alpha=0.5)
    ax2.axvline(expected_magnitude + magnitude_uncertainty, color='r', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Magnitude', fontsize=10)
    ax2.set_ylabel('Probability Density', fontsize=10)
    ax2.set_title('Expected Magnitude Distribution', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add magnitude annotations
    ax2.annotate(f'Mean: {expected_magnitude:.4f}',
                 xy=(expected_magnitude, ax2.get_ylim()[1]),
                 xytext=(0, 5),
                 textcoords='offset points',
                 ha='center',
                 fontsize=9)

    plt.tight_layout()


def plot_trend_magnitude_duration_heatmap(trace, current_trend, max_duration=30, magnitude_steps=10):
    """Plot heatmap of trend probabilities with magnitude vs duration"""
    # Create grid of durations and magnitudes
    durations = np.arange(1, max_duration + 1)

    # Calculate magnitude range based on historical data
    base_magnitude = current_trend['magnitude']
    magnitude_range = np.linspace(base_magnitude * 0.5, base_magnitude * 2, magnitude_steps)
    probabilities = np.zeros((len(magnitude_range), len(durations)))

    # Extract posterior distributions
    if current_trend['direction'] == 1:
        duration_samples = trace.posterior['up_duration_mu'].values.flatten()
    else:
        duration_samples = trace.posterior['down_duration_mu'].values.flatten()

    magnitude_mu = np.mean(trace.posterior['magnitude_mu'].values.flatten())
    magnitude_sigma = np.mean(trace.posterior['magnitude_sigma'].values.flatten())

    duration_mu = np.mean(duration_samples)
    duration_sigma = np.std(duration_samples)

    # Calculate joint probabilities for each combination
    for i, mag in enumerate(magnitude_range):
        for j, dur in enumerate(durations):
            # Combine duration and magnitude probabilities
            dur_z_score = (dur - duration_mu) / (duration_sigma + 1e-6)
            mag_z_score = (mag - magnitude_mu) / (magnitude_sigma + 1e-6)

            # Joint probability (assuming independence)
            dur_prob = 1 - stats.norm.cdf(dur_z_score)
            mag_prob = stats.norm.pdf(mag_z_score) / stats.norm.pdf(0)  # Normalize to peak at 1

            probabilities[i, j] = max(0.1, min(0.9, dur_prob * mag_prob))

    # Create heatmap
    plt.figure(figsize=(12, 8))

    # Plot heatmap
    im = plt.imshow(probabilities,
                    aspect='auto',
                    origin='lower',
                    extent=[0, max_duration, magnitude_range[0], magnitude_range[-1]],
                    cmap='RdYlBu_r')

    # Add colorbar
    plt.colorbar(im, label='Joint Probability')

    # Mark current trend
    plt.axvline(x=current_trend['duration'],
                color='black',
                linestyle='--',
                alpha=0.5,
                label=f'Current Duration ({current_trend["duration"]} days)')
    plt.axhline(y=current_trend['magnitude'],
                color='black',
                linestyle=':',
                alpha=0.5,
                label=f'Current Magnitude ({current_trend["magnitude"]:.4f})')

    # Customize plot
    plt.title(
        f'Trend Duration-Magnitude Probability Heatmap\n{"Uptrend" if current_trend["direction"] == 1 else "Downtrend"}',
        pad=20)
    plt.xlabel('Duration (days)')
    plt.ylabel('Magnitude')
    plt.legend()

    # Add grid
    plt.grid(False)

    plt.tight_layout()


# ======== Main Execution ========
def main():
    # Fetch data
    prices = fetch_market_data()
    if prices is None:
        return

    # Perform wavelet analysis
    print("\n🔍 Performing wavelet transform analysis...")
    coefficients, periods, normalized_data = perform_wavelet_analysis(prices)
    top_periods, mean_power = identify_dominant_cycles(coefficients, periods)

    print("\n📊 Dominant Market Cycles:")
    for i, period in enumerate(top_periods):
        print(f"   Cycle {i + 1}: {period:.1f} trading days")

    # Extract historical trends
    print("\n🔍 Extracting historical trend patterns...")
    trends = extract_trends(prices)

    if len(trends) < 5:
        print("⚠️ Insufficient trend data for Bayesian analysis")
        print("\n📈 Displaying available trend information...")

        # Still show available trends
        plot_trends_on_price(prices, trends)
        plot_wavelet_analysis(coefficients, periods, prices)

        if trends:  # If we have any trends at all
            current_trend = trends[-1]
            trend_type = "Uptrend" if current_trend['direction'] == 1 else "Downtrend"
            print(f"\nCurrent Trend Information:")
            print(f"   Type: {trend_type}")
            print(f"   Started: {current_trend['start'].date()}")
            print(f"   Duration: {current_trend['duration']} days")
            print(f"   Strength: {current_trend['strength']:.2f}")
            print(f"   Magnitude: {current_trend['magnitude']:.4f}")

        plt.show()
        return

    # Identify current trend
    current_trend = trends[-1]
    trend_type = "Uptrend" if current_trend['direction'] == 1 else "Downtrend"
    print(f"\n📈 Current Trend: {trend_type}")
    print(f"   Started: {current_trend['start'].date()}")
    print(f"   Duration: {current_trend['duration']} days")
    print(f"   Magnitude: {current_trend['magnitude']:.4f}")

    # Build Bayesian model
    print("\n🧮 Building Bayesian model...")
    trace = build_bayesian_model(trends)

    # Calculate trend probabilities
    continuation_probs, expected_magnitude, magnitude_uncertainty = calculate_trend_probabilities(trace, current_trend)

    print("\n🔮 Trend Continuation Probabilities:")
    for days, prob in continuation_probs.items():
        print(f"   Next {days} days: {prob:.2f}")

    print(f"\n📏 Expected Magnitude Change: {expected_magnitude:.4f} ± {magnitude_uncertainty:.4f}")

    # Visualize results
    print("\n🎨 Generating visualizations...")
    plot_trends_on_price(prices, trends)
    plot_wavelet_analysis(coefficients, periods, prices)
    plot_bayesian_results(trace, trends, current_trend, continuation_probs)
    plot_trend_probabilities(continuation_probs, expected_magnitude, magnitude_uncertainty)
    plot_trend_magnitude_duration_heatmap(trace, current_trend)
    plt.show()


if __name__ == "__main__":
    main()
