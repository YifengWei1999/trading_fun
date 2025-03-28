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
from statsmodels.tsa.seasonal import seasonal_decompose
import calendar
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

# ======== Configuration ========
SYMBOL = '^TNX'  # 10-Year Treasury Yield
START_DATE = '2023-01-01'  # Extended history for better wavelet analysis
END_DATE = '2025-03-28'
# END_DATE = datetime.now().strftime('%Y-%m-%d')
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
def build_bayesian_model(trends, seasonal_features=None, dominant_cycles=None):
    """Enhanced Bayesian model with support for oscillating markets"""
    # Split trends by type
    trending_trends = [t for t in trends if t.get('type', 'trending') == 'trending']
    oscillating_trends = [t for t in trends if t.get('type', '') == 'oscillating']

    # Process trend data
    durations = np.array([trend['duration'] for trend in trending_trends])
    magnitudes = np.array([trend['magnitude'] for trend in trending_trends])
    directions = np.array([trend['direction'] for trend in trending_trends])

    # Additional data for oscillating markets
    if oscillating_trends:
        osc_durations = np.array([trend['duration'] for trend in oscillating_trends])
        osc_ranges = np.array([trend['magnitude'] for trend in oscillating_trends])
    else:
        # Default values if no oscillating periods found
        osc_durations = np.array([7.0])  # Default 7-day oscillation
        osc_ranges = np.array([0.05])  # Default 5bp range

    # Filter to valid ranges and add small jitter to prevent exact zeros
    min_duration = 1
    durations = np.maximum(durations, min_duration)

    # Create separate arrays for up and down trends
    up_durations = np.array([d for d, direction in zip(durations, directions) if direction == 1])
    down_durations = np.array([d for d, direction in zip(durations, directions) if direction == -1])

    # Ensure we have at least one data point for each direction
    if len(up_durations) == 0:
        up_durations = np.array([min_duration])
    if len(down_durations) == 0:
        down_durations = np.array([min_duration])

    # Calculate empirical statistics
    up_mean = np.mean(up_durations)
    up_std = max(1.0, np.std(up_durations))
    down_mean = np.mean(down_durations)
    down_std = max(1.0, np.std(down_durations))
    mag_mean = np.mean(magnitudes)
    mag_std = max(0.1, np.std(magnitudes))

    # Oscillation statistics
    osc_mean = np.mean(osc_durations)
    osc_std = max(1.0, np.std(osc_durations))
    range_mean = np.mean(osc_ranges)
    range_std = max(0.01, np.std(osc_ranges))

    with pm.Model() as model:
        # Standard trend parameters
        up_mu = pm.Normal('up_duration_mu', mu=up_mean, sigma=2)
        up_sigma = pm.HalfNormal('up_duration_sigma', sigma=2)

        down_mu = pm.Normal('down_duration_mu', mu=down_mean, sigma=2)
        down_sigma = pm.HalfNormal('down_duration_sigma', sigma=2)

        # Add oscillation parameters
        osc_mu = pm.Normal('oscillation_duration_mu', mu=osc_mean, sigma=2)
        osc_sigma = pm.HalfNormal('oscillation_duration_sigma', sigma=2)

        range_mu = pm.Normal('oscillation_range_mu', mu=range_mean, sigma=range_std)
        range_sigma = pm.HalfNormal('oscillation_range_sigma', sigma=range_std / 2)

        # Magnitude parameters
        mag_mu = pm.Normal('magnitude_mu', mu=mag_mean, sigma=mag_std)
        mag_sigma = pm.HalfNormal('magnitude_sigma', sigma=mag_std / 2)

        # Correlation parameter
        rho = pm.Normal('duration_magnitude_correlation', mu=0, sigma=0.3)

        # Likelihoods
        pm.Normal('up_duration', mu=up_mu, sigma=up_sigma, observed=up_durations)
        pm.Normal('down_duration', mu=down_mu, sigma=down_sigma, observed=down_durations)
        pm.Normal('magnitude', mu=mag_mu, sigma=mag_sigma, observed=magnitudes)

        # Oscillation likelihoods
        pm.Normal('oscillation_duration', mu=osc_mu, sigma=osc_sigma, observed=osc_durations)
        pm.Normal('oscillation_range', mu=range_mu, sigma=range_sigma, observed=osc_ranges)

        # Fit the model
        approx = pm.fit(
            n=5000,
            method='advi',
            obj_optimizer=pm.adam(learning_rate=0.03),
            callbacks=[
                pm.callbacks.CheckParametersConvergence(tolerance=0.02, diff='relative')
            ]
        )
        trace = approx.sample(1000)

    return trace


def calculate_indicators(data, wavelet_info=None):
    """Calculate technical indicators with wavelet cycle information"""
    indicators = {}

    # Standard indicators
    # Multiple timeframe EMAs
    timeframes = {
        'short': 5,  # Very short-term
        'medium': 13,  # Short-term
        'long': 34  # Medium-term
    }

    # Calculate EMAs for each timeframe
    for tf_name, period in timeframes.items():
        indicators[f'ema_{tf_name}'] = data.ewm(span=period, adjust=False).mean()

    # Calculate rate of change (momentum)
    indicators['roc_short'] = data.pct_change(periods=5)
    indicators['roc_medium'] = data.pct_change(periods=13)

    # Calculate volatility
    indicators['volatility'] = data.rolling(window=21).std() / data

    # Calculate ADX-like trend strength
    high_low_diff = data.rolling(window=14).max() - data.rolling(window=14).min()
    indicators['trend_strength'] = (high_low_diff / data.rolling(window=14).mean()).rolling(window=14).mean()

    # Add wavelet-based cycle indicators if available
    if wavelet_info is not None:
        wavelet_coefs, wavelet_periods = wavelet_info

        # Additional wavelet-based indicators would go here

    return indicators


def identify_trend_changes(data, indicators, wavelet_info=None, min_trend_duration=2):
    """Identify trend changes with support for mean-reverting/oscillating market regimes"""
    trends = []
    trend_start = None
    current_direction = 0  # 0=neutral, 1=up, -1=down, 2=oscillating/mean-reverting

    # Parameters for trend detection
    confirmation_threshold = 2
    potential_change_count = 0
    potential_new_direction = 0
    oscillation_window = 10  # Days to consider for oscillation detection
    oscillation_threshold = 0.6  # Threshold for oscillation score

    for i in range(34, len(data)):  # Start after all indicators are valid
        # Get current indicator values
        ema_short = indicators['ema_short'].iloc[i]  # Use .iloc instead of []
        ema_medium = indicators['ema_medium'].iloc[i]  # Use .iloc instead of []
        ema_long = indicators['ema_long'].iloc[i]  # Use .iloc instead of []

        # Check for oscillating/mean-reverting behavior
        if i >= oscillation_window + 34:
            # Look at recent price action
            recent_window = data[i - oscillation_window:i + 1]

            # Calculate oscillation indicators
            # Fix indices in the slices as well
            ema_short_slice = indicators['ema_short'][i - oscillation_window:i + 1]
            ema_medium_slice = indicators['ema_medium'][i - oscillation_window:i + 1]
            ema_crossovers = sum(np.diff(np.sign(ema_short_slice - ema_medium_slice)) != 0)

            price_range = (recent_window.max() - recent_window.min()) / recent_window.mean()

            # Mean reversion can be detected by price repeatedly returning to its mean
            distance_from_mean = abs(recent_window.iloc[-1] - recent_window.mean()) / recent_window.std()

            # Calculate an oscillation score (high = oscillating, low = trending)
            oscillation_score = (ema_crossovers / oscillation_window) * (1 - distance_from_mean)

            # Detect if we're in a mean-reverting/oscillating regime
            is_oscillating = oscillation_score > oscillation_threshold
        else:
            is_oscillating = False

        # Determine trend direction with support for oscillation
        if is_oscillating:
            new_direction_candidate = 2  # Oscillating/mean-reverting
        elif (ema_short > ema_medium and ema_medium > ema_long):
            new_direction_candidate = 1  # Up trend
        elif (ema_short < ema_medium and ema_medium < ema_long):
            new_direction_candidate = -1  # Down trend
        else:
            # For rates, use short-term EMA crossing for faster detection
            if ema_short > ema_medium:
                new_direction_candidate = 1
            elif ema_short < ema_medium:
                new_direction_candidate = -1
            else:
                new_direction_candidate = current_direction

        # Handle potential changes with new oscillation state
        if new_direction_candidate != current_direction:
            # For oscillation, require less confirmation
            required_threshold = 1 if new_direction_candidate == 2 else confirmation_threshold

            if potential_new_direction == new_direction_candidate:
                potential_change_count += 1
            else:
                potential_change_count = 1
                potential_new_direction = new_direction_candidate

            if potential_change_count >= required_threshold:
                new_direction = new_direction_candidate
                potential_change_count = 0
            else:
                new_direction = current_direction
        else:
            new_direction = current_direction
            potential_change_count = 0

        # Process trend changes
        if new_direction != current_direction:
            # Record previous trend
            if trend_start is not None and i > 0:
                segment = data.loc[trend_start:data.index[i - 1]]

                if len(segment) >= min_trend_duration:
                    # Calculate trend metrics
                    price_change_bps = (segment.iloc[-1] - segment.iloc[0]) * 100  # Use iloc instead of []

                    # For oscillating trends, direction is 0 and we use different metrics
                    if current_direction == 2:
                        direction = 0  # Neutral/oscillating
                        # For oscillating periods, measure range instead of direction
                        range_bps = (segment.max() - segment.min()) * 100
                        strength = oscillation_score  # Use oscillation score as strength
                        magnitude = range_bps / 100  # Keep in decimal
                    else:
                        direction = 1 if price_change_bps > 0 else -1
                        magnitude = abs(price_change_bps / 100)

                        # Calculate strength based on price movement vs volatility
                        mom_avg = np.mean(indicators['roc_medium'].loc[segment.index]) * 100
                        vol_avg = np.mean(indicators['volatility'].loc[segment.index])
                        strength = abs(mom_avg) / (vol_avg + 0.001)  # Avoid div by zero

                    trends.append({
                        'start': trend_start,
                        'end': data.index[i - 1],
                        'duration': (data.index[i - 1] - trend_start).days,
                        'magnitude': magnitude,
                        'direction': direction,
                        'strength': strength,
                        'start_price': segment.iloc[0],  # Use iloc instead of []
                        'end_price': segment.iloc[-1],  # Use iloc instead of []
                        'volatility': np.mean(indicators['volatility'][segment.index]),
                        'is_recent': (data.index[i - 1] >= data.index[-60]),
                        'type': 'oscillating' if current_direction == 2 else 'trending'
                    })

            # Start new trend
            trend_start = data.index[i]
            current_direction = new_direction

    # Handle final trend segment
    if trend_start is not None:
        segment = data.loc[trend_start:]

        if len(segment) >= 2:
            price_change_bps = (segment.iloc[-1] - segment.iloc[0]) * 100  # Use iloc instead of []

            # For oscillating trends, use different metrics
            if current_direction == 2:
                direction = 0  # Neutral/oscillating
                range_bps = (segment.max() - segment.min()) * 100
                magnitude = range_bps / 100
                strength = 0.5  # Default strength for final oscillating segment
            else:
                direction = 1 if price_change_bps > 0 else -1
                magnitude = abs(price_change_bps / 100)

                # Calculate strength
                mom_avg = np.mean(indicators['roc_medium'].loc[segment.index]) * 100
                vol_avg = np.mean(indicators['volatility'].loc[segment.index])
                strength = abs(mom_avg) / (vol_avg + 0.001)

            trends.append({
                'start': trend_start,
                'end': data.index[-1],
                'duration': (data.index[-1] - trend_start).days,
                'magnitude': magnitude,
                'direction': direction,
                'strength': strength,
                'start_price': segment.iloc[0],  # Use iloc instead of []
                'end_price': segment.iloc[-1],  # Use iloc instead of []
                'volatility': np.mean(indicators['volatility'][segment.index]),
                'is_recent': True,
                'type': 'oscillating' if current_direction == 2 else 'trending'
            })

    return trends


def calculate_trend_probabilities(current_trend, trace, max_days=10):
    """
    Calculate probability of trend continuation for various time horizons
    """
    probabilities = {}

    # Get relevant posterior distributions
    if current_trend.get('type', '') == 'oscillating':
        # For oscillating markets, use oscillation duration distribution
        if 'oscillation_duration_mu' in trace.posterior:
            duration_samples = trace.posterior['oscillation_duration_mu'].values.flatten()
        else:
            # Fallback if oscillation parameters don't exist
            duration_samples = np.array([7.0])  # Default value
    else:
        # For directional trends, use the appropriate direction
        if current_trend['direction'] == 1:
            duration_samples = trace.posterior['up_duration_mu'].values.flatten()
        else:
            duration_samples = trace.posterior['down_duration_mu'].values.flatten()

    # Calculate how long the current trend has been ongoing
    current_duration = current_trend['duration']

    # Calculate continuation probabilities for different time horizons
    days_horizons = [1, 2, 3, 5, 10]
    days_horizons = [d for d in days_horizons if d <= max_days]

    for days in days_horizons:
        # Probability = P(total_duration > current_duration + days | total_duration > current_duration)
        # This is a conditional probability
        prob_continue_beyond_current = np.mean(duration_samples > current_duration)
        prob_continue_beyond_target = np.mean(duration_samples > (current_duration + days))

        if prob_continue_beyond_current > 0:
            conditional_prob = prob_continue_beyond_target / prob_continue_beyond_current
        else:
            conditional_prob = 0.2  # Default fallback probability

        # Adjust for trend strength
        trend_strength = min(1.0, max(0.2, current_trend.get('strength', 0.5)))
        adjusted_prob = conditional_prob * trend_strength

        # Ensure probability is between 0 and 1
        probabilities[days] = max(0.1, min(0.9, adjusted_prob))

    return probabilities


def add_volatility_regimes(prices, trends):
    """Add volatility regime information to trend data"""
    # Calculate rolling volatility on price data (annualized)
    rolling_vol = prices.pct_change().rolling(window=21).std() * np.sqrt(252)

    # Fill any NaN values with median
    rolling_vol = rolling_vol.fillna(rolling_vol.median())

    # Determine overall volatility characteristics
    median_vol = rolling_vol.median()
    high_vol_threshold = rolling_vol.quantile(0.7)  # 70th percentile
    low_vol_threshold = rolling_vol.quantile(0.3)  # 30th percentile

    # Add volatility regime to each trend
    for i, trend in enumerate(trends):
        # Get volatility during this trend period
        start_date = trend['start']
        end_date = trend['end']

        # Make sure the dates are in the index
        if start_date in rolling_vol.index and end_date in rolling_vol.index:
            trend_vol = rolling_vol.loc[start_date:end_date].mean()
        else:
            # If dates not in index, use the closest available dates
            trend_vol = median_vol

        # Classify regime
        if trend_vol > high_vol_threshold:
            trends[i]['vol_regime'] = 'high'
        elif trend_vol < low_vol_threshold:
            trends[i]['vol_regime'] = 'low'
        else:
            trends[i]['vol_regime'] = 'medium'

        # Store the actual volatility value
        trends[i]['volatility_value'] = trend_vol

    return trends, median_vol, high_vol_threshold, low_vol_threshold


def generate_trading_signals(trends, trace, prices, volatility_info=None):
    """Generate trading signals based on trend analysis and Bayesian forecasts"""
    # Create a DataFrame with the same index as prices
    signals = pd.DataFrame(index=prices.index)
    signals['yield'] = prices
    signals['position'] = 0  # 1 = long rates (short bonds), -1 = short rates (long bonds), 0 = flat
    signals['signal_strength'] = 0.0  # Confidence level (0-1)
    signals['regime'] = 'unknown'  # Market regime
    signals['vol_regime'] = 'medium'  # Volatility regime

    # For each trend period, assign positions
    for trend in trends:
        start = trend['start']
        end = trend['end']

        # Skip if dates outside our data range
        if start not in signals.index or end not in signals.index:
            continue

        # Get date range for this trend
        trend_dates = prices.loc[start:end].index
        if len(trend_dates) == 0:
            continue

        # Determine position based on trend direction and type
        if trend.get('type', '') == 'oscillating':
            # For oscillating markets, use mean-reversion strategy
            period_data = prices.loc[start:end]
            if len(period_data) >= 3:  # Need at least 3 points for mean calculation
                period_mean = period_data.mean()
                period_std = period_data.std()

                for date in trend_dates:
                    if date in period_data.index:
                        deviation = (period_data[date] - period_mean) / (period_std + 1e-8)

                        # Mean-reversion: go opposite deviation from mean
                        if deviation > 0.8:  # Significantly above mean
                            signals.loc[date, 'position'] = -1  # Short rates
                            signals.loc[date, 'signal_strength'] = min(0.8, abs(deviation) / 2)
                        elif deviation < -0.8:  # Significantly below mean
                            signals.loc[date, 'position'] = 1  # Long rates
                            signals.loc[date, 'signal_strength'] = min(0.8, abs(deviation) / 2)
                        else:
                            signals.loc[date, 'position'] = 0  # Neutral
                            signals.loc[date, 'signal_strength'] = 0.0

                        signals.loc[date, 'regime'] = 'oscillating'
        else:
            # For trending markets, align with the trend
            direction = trend['direction']
            strength = min(1.0, max(0.1, trend.get('strength', 0.5)))

            for date in trend_dates:
                # Simple trend-following strategy
                signals.loc[date, 'position'] = direction
                signals.loc[date, 'signal_strength'] = strength
                signals.loc[date, 'regime'] = 'uptrend' if direction == 1 else 'downtrend'

        # Add volatility regime information if available
        if 'vol_regime' in trend:
            signals.loc[trend_dates, 'vol_regime'] = trend['vol_regime']

    # Adjust position sizing based on volatility regime
    for date in signals.index:
        vol_regime = signals.loc[date, 'vol_regime']

        # Scale position size by volatility
        if vol_regime == 'high':
            signals.loc[date, 'signal_strength'] *= 0.7  # Reduce position in high vol
        elif vol_regime == 'low':
            signals.loc[date, 'signal_strength'] *= 1.3  # Increase position in low vol

        # Ensure signal strength stays in [0,1] range
        signals.loc[date, 'signal_strength'] = min(1.0, max(0.0, signals.loc[date, 'signal_strength']))

    # Calculate final position size
    signals['position_size'] = signals['position'] * signals['signal_strength']

    return signals.fillna(0)  # Fill any NaN values with zeros


def backtest_strategy_walk_forward(prices, risk_per_trade=1000, initial_window=252, step_size=21):
    """Improved walk-forward backtest with enhanced signal generation including reversals"""
    # Initial setup remains the same
    results = pd.DataFrame(index=prices.index)
    results['yield'] = prices
    results['position'] = 0
    results['signal_strength'] = 0.0
    results['regime'] = 'unknown'
    results['vol_regime'] = 'medium'

    start_idx = initial_window
    trades = []

    print(f"Starting walk-forward analysis with {initial_window} days initial window")

    # Loop through the data in steps
    while start_idx < len(prices):
        # Define current window end
        end_idx = min(start_idx + step_size, len(prices))

        print(
            f"Processing period {prices.index[start_idx].date()} to {prices.index[min(end_idx - 1, len(prices) - 1)].date()}")

        # Define training data
        train_prices = prices.iloc[:start_idx]

        # Analysis on training data
        coefficients, periods, normalized_data = perform_wavelet_analysis(train_prices)
        indicators = calculate_indicators(train_prices, (coefficients, periods))
        trends = identify_trend_changes(train_prices, indicators, (coefficients, periods), min_trend_duration=2)
        trends, _, _, _ = add_volatility_regimes(train_prices, trends)

        # Only proceed if we have trends
        if trends:
            # Build Bayesian model
            trace = build_bayesian_model(trends)

            # Get the current trend
            current_trend = trends[-1]

            # Calculate continuation probabilities
            continuation_probs = calculate_trend_probabilities(current_trend, trace, max_days=step_size)

            # Debug
            print(
                f"  Current trend: {'Up' if current_trend.get('direction', 0) == 1 else 'Down' if current_trend.get('direction', 0) == -1 else 'Oscillating'}, strength: {current_trend.get('strength', 0):.2f}")
            if continuation_probs:
                next_days = min(5, max(continuation_probs.keys()))
                prob = continuation_probs.get(next_days, 0)
                print(f"  Continuation probability (next {next_days} days): {prob:.2f}")

                # Add reversal probability info
                if prob < 0.3:
                    print(f"  ⚠️ Low continuation probability - potential reversal signal!")

            # Generate signals for the test period
            test_prices = prices.iloc[start_idx:end_idx]

            if not test_prices.empty:
                test_signals = pd.DataFrame(index=test_prices.index)
                test_signals['yield'] = test_prices

                # IMPROVED SIGNAL GENERATION:
                if current_trend.get('type', '') == 'oscillating':
                    # For oscillating markets, use mean reversion with more nuance
                    # Calculate distance from recent mean
                    recent_window = train_prices.iloc[-20:]  # Last 20 days
                    recent_mean = recent_window.mean()
                    recent_std = recent_window.std()

                    for date in test_signals.index:
                        # Calculate z-score from mean
                        z_score = (test_signals.loc[date, 'yield'] - recent_mean) / recent_std

                        # Only take strong mean-reversion signals
                        if z_score > 1.0:  # Above mean - go short
                            test_signals.loc[date, 'position'] = -1
                            test_signals.loc[date, 'signal_strength'] = min(0.8, abs(z_score) / 3)
                        elif z_score < -1.0:  # Below mean - go long
                            test_signals.loc[date, 'position'] = 1
                            test_signals.loc[date, 'signal_strength'] = min(0.8, abs(z_score) / 3)
                        else:
                            test_signals.loc[date, 'position'] = 0
                            test_signals.loc[date, 'signal_strength'] = 0

                        test_signals.loc[date, 'regime'] = 'oscillating'
                else:
                    # For trending markets, use probability-adjusted signals WITH REVERSAL LOGIC
                    direction = current_trend.get('direction', 0)
                    strength = min(1.0, max(0.1, current_trend.get('strength', 0.5)))

                    # Get short-term probability for dynamic position sizing
                    short_prob = 0.5
                    for days in [1, 2, 3, 5]:
                        if days in continuation_probs:
                            short_prob = continuation_probs[days]
                            break

                    # NEW REVERSAL LOGIC:
                    if short_prob > 0.55:  # Strong continuation signal
                        # Follow the trend with confidence
                        position = direction
                        # Adjust strength by probability (higher probability = larger position)
                        prob_adjustment = (short_prob - 0.5) * 2
                        adjusted_strength = strength * prob_adjustment
                        regime_label = 'strong_' + ('uptrend' if direction == 1 else 'downtrend')

                    elif short_prob < 0.30:  # Strong reversal signal
                        # Take contrarian position - bet on trend reversal
                        position = -direction  # Opposite of the current trend
                        # Adjust strength by inverse probability (lower continuation = stronger reversal)
                        prob_adjustment = (0.5 - short_prob) * 2
                        adjusted_strength = strength * prob_adjustment
                        regime_label = 'reversal_signal'

                    elif short_prob < 0.45:  # Weak continuation, potential early reversal
                        # Take smaller contrarian position
                        position = -direction
                        prob_adjustment = (0.45 - short_prob) * 3  # Less aggressive scaling
                        adjusted_strength = strength * prob_adjustment * 0.7  # Reduce size for early signals
                        regime_label = 'potential_reversal'

                    else:  # Uncertain (0.45-0.55)
                        # Stay neutral or take very small position in trend direction
                        position = 0
                        adjusted_strength = 0
                        regime_label = 'uncertain'

                    # Assign values
                    test_signals['position'] = position
                    test_signals['signal_strength'] = adjusted_strength
                    test_signals['regime'] = regime_label

                # Apply volatility adjustments
                vol_regime = current_trend.get('vol_regime', 'medium')
                test_signals['vol_regime'] = vol_regime

                # More conservative volatility adjustments
                if vol_regime == 'high':
                    test_signals['signal_strength'] *= 0.5  # Reduced from 0.7 - more conservative
                elif vol_regime == 'low':
                    test_signals['signal_strength'] *= 1.2  # Reduced from 1.3 - more conservative

                # Apply filter for small positions (reduce trading noise)
                test_signals.loc[test_signals['signal_strength'] < 0.15, 'position'] = 0
                test_signals.loc[test_signals['signal_strength'] < 0.15, 'signal_strength'] = 0

                # Ensure signal strength stays in range
                test_signals['signal_strength'] = test_signals['signal_strength'].clip(0, 1)

                # Calculate position size
                test_signals['position_size'] = test_signals['position'] * test_signals['signal_strength']

                # Update results DataFrame
                for col in ['position', 'signal_strength', 'regime', 'vol_regime', 'position_size']:
                    if col in test_signals.columns:
                        results.loc[test_signals.index, col] = test_signals[col]

        # Move to next window
        start_idx = end_idx

    # Calculate yield changes in basis points
    results['yield_change_bp'] = results['yield'].diff() * 100

    # For rates, we use a simplified DV01 approximation
    results['dv01'] = 850  # Simplified fixed DV01 for 10Y Treasury

    # Calculate position size in DV01 terms
    results['dv01_position'] = risk_per_trade / results['dv01'] * results['position_size']

    # Calculate daily P&L
    results['daily_pnl'] = results['dv01_position'].shift(1) * results['yield_change_bp']

    # Drop NaN values from P&L calculation
    results = results.dropna(subset=['daily_pnl'])

    # Calculate cumulative P&L and drawdowns
    results['cumulative_pnl'] = results['daily_pnl'].cumsum()
    results['peak'] = results['cumulative_pnl'].cummax()
    results['drawdown'] = results['peak'] - results['cumulative_pnl']

    # Calculate rolling Sharpe ratio (63 days = approximately 3 months)
    window = min(63, len(results) // 2)  # Use shorter window if not enough data
    if window > 5:  # Only calculate if we have enough data
        results['rolling_ret'] = results['daily_pnl'].rolling(window=window).mean()
        results['rolling_vol'] = results['daily_pnl'].rolling(window=window).std()
        results['rolling_sharpe'] = results['rolling_ret'] / results['rolling_vol'] * np.sqrt(252)
    else:
        # If not enough data, create a placeholder column
        results['rolling_sharpe'] = 0.0

    # Calculate performance metrics
    total_return = results['daily_pnl'].sum()
    win_rate = (results['daily_pnl'] > 0).mean()
    max_drawdown = results['drawdown'].max()

    # Only calculate Sharpe if we have sufficient data
    if len(results) > 10:
        sharpe_ratio = (results['daily_pnl'].mean() / results['daily_pnl'].std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Count trades
    position_changes = results['position'] != results['position'].shift(1)
    total_trades = position_changes.sum()

    # Create trade details
    current_position = 0
    entry_date = None
    entry_yield = None

    # Loop through signals to identify trades
    for date, row in results.iterrows():
        if row['position'] != current_position and current_position != 0:
            # Exit of a position
            if entry_date is not None:
                trade = {
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_yield': entry_yield,
                    'exit_yield': row['yield'],
                    'position': 'Long Rates' if current_position > 0 else 'Short Rates',
                    'duration': (date - entry_date).days,
                    'pnl': (row['yield'] - entry_yield) * row['dv01'] * current_position
                }
                trades.append(trade)

            entry_date = None
            entry_yield = None

        if row['position'] != current_position and row['position'] != 0:
            # Entry of a new position
            entry_date = date
            entry_yield = row['yield']

        current_position = row['position']

    # Create performance summary
    performance = {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'avg_trade_pnl': total_return / max(1, total_trades),
        'avg_trade_duration': len(results) / max(1, total_trades),
        'profitable_trades_pct': win_rate * 100
    }

    # Add performance summary to main results
    print("\n💰 Strategy Performance:")
    print(f"   Total Return: ${performance['total_return']:.2f}")
    print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: ${performance['max_drawdown']:.2f}")
    print(f"   Win Rate: {performance['win_rate'] * 100:.1f}%")
    print(f"   Total Trades: {performance['total_trades']}")

    return results, performance, trades


def plot_backtest_results(results, performance, trades):
    """Plot backtest results with defensive checks for missing data"""
    # Create figure with grid layout
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])

    # Plot 1: Price chart with positions
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(results.index, results['yield'], 'k-', linewidth=1.5)

    # Color background based on position
    for i in range(1, len(results)):
        if results['position'].iloc[i] > 0:  # Long rates
            ax1.axvspan(results.index[i - 1], results.index[i], alpha=0.2, color='green')
        elif results['position'].iloc[i] < 0:  # Short rates
            ax1.axvspan(results.index[i - 1], results.index[i], alpha=0.2, color='red')

    # Mark regime changes
    if 'regime' in results.columns:
        regime_changes = results[results['regime'] != results['regime'].shift(1)].index
        for date in regime_changes:
            if date != results.index[0]:  # Skip the first point
                regime = results.loc[date, 'regime']
                color = 'green' if regime == 'uptrend' else 'red' if regime == 'downtrend' else 'purple'
                ax1.axvline(date, color=color, linestyle='--', alpha=0.7)

    ax1.set_title('10-Year Treasury Yield with Trading Positions', fontsize=14)
    ax1.set_ylabel('Yield (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.2, label='Long Rates (Short Bonds)'),
        Patch(facecolor='red', alpha=0.2, label='Short Rates (Long Bonds)'),
        plt.Line2D([0], [0], color='k', linewidth=1.5, label='10Y Yield')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')

    # Plot 2: Cumulative P&L
    ax2 = plt.subplot(gs[1, :], sharex=ax1)
    ax2.plot(results.index, results['cumulative_pnl'], 'b-', linewidth=1.5)
    ax2.fill_between(results.index, 0, results['cumulative_pnl'], alpha=0.3, color='blue')

    # Mark drawdowns
    drawdown_threshold = performance['max_drawdown'] * 0.5
    significant_drawdowns = results[results['drawdown'] > drawdown_threshold]
    for date in significant_drawdowns.index:
        ax2.plot(date, results.loc[date, 'cumulative_pnl'], 'ro', alpha=0.7, markersize=4)

    ax2.set_title('Cumulative P&L', fontsize=14)
    ax2.set_ylabel('P&L ($)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Rolling Sharpe Ratio (if available)
    ax3 = plt.subplot(gs[2, 0], sharex=ax1)
    if 'rolling_sharpe' in results.columns and results['rolling_sharpe'].notna().any():
        ax3.plot(results.index, results['rolling_sharpe'], 'g-', linewidth=1.5)
        ax3.axhline(y=1, color='r', linestyle='--', alpha=0.7)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.7)
        ax3.set_title('Rolling 3-Month Sharpe Ratio', fontsize=14)
    else:
        # Create placeholder text if data not available
        ax3.text(0.5, 0.5, "Rolling Sharpe Ratio\n(insufficient data)",
                 ha='center', va='center', fontsize=12)
        ax3.set_axis_off()

    ax3.set_ylabel('Sharpe Ratio', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Position Size
    ax4 = plt.subplot(gs[2, 1], sharex=ax1)
    ax4.fill_between(results.index, 0, results['position_size'],
                     where=results['position_size'] > 0, color='green', alpha=0.5)
    ax4.fill_between(results.index, 0, results['position_size'],
                     where=results['position_size'] < 0, color='red', alpha=0.5)

    ax4.set_title('Position Size (Scaled by Signal Strength)', fontsize=14)
    ax4.set_ylabel('Position', fontsize=12)
    ax4.grid(True, alpha=0.3)

    # Add performance metrics as text
    metrics_text = (
        f"Total Return: ${performance['total_return']:.2f}\n"
        f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: ${performance['max_drawdown']:.2f}\n"
        f"Win Rate: {performance['win_rate'] * 100:.1f}%\n"
        f"Total Trades: {performance['total_trades']}\n"
        f"Avg Trade P&L: ${performance['avg_trade_pnl']:.2f}\n"
        f"Avg Trade Duration: {performance['avg_trade_duration']:.1f} days\n"
        f"Profitable Trades: {performance['profitable_trades_pct']:.1f}%"
    )

    fig.text(0.02, 0.02, metrics_text, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    plt.tight_layout()
    return fig


# ======== Main Execution ========
def main():
    """Main execution function"""
    # Fetch market data
    prices = fetch_market_data()
    if prices is None:
        return

    # Perform wavelet analysis for visualization purposes
    print("\n🔍 Extracting historical trend patterns with cycle analysis...")
    coefficients, periods, normalized_data = perform_wavelet_analysis(prices)

    # Identify dominant cycles
    top_periods, mean_power = identify_dominant_cycles(coefficients, periods)

    print("\n📊 Dominant market cycles:")
    for i, period in enumerate(top_periods[:3], 1):
        print(f"   Cycle {i}: {period:.1f} trading days")

    # Extract historical trends (for current analysis only, not backtesting)
    trends = extract_trends(prices, coefficients=coefficients, periods=periods)

    # Add volatility regime detection
    trends, median_vol, high_vol_threshold, low_vol_threshold = add_volatility_regimes(prices, trends)

    # Analyze seasonality
    seasonal_results = analyze_seasonality(prices)

    # Run backtest with proper walk-forward analysis
    print("\n📊 Running walk-forward strategy backtest (no look-ahead bias)...")
    results, performance, trades = backtest_strategy_walk_forward(prices, risk_per_trade=1000,
                                                                  initial_window=252, step_size=21)

    # Fit final model for current analysis and forecasting
    print("\n🔄 Fitting probabilistic model to historical trends...")
    trace = build_bayesian_model(trends, seasonal_features=seasonal_results)

    # Get the current trend
    current_trend = trends[-1]
    if 'type' in current_trend and current_trend['type'] == 'oscillating':
        print(f"\n📈 Current Market: Oscillating/Mean-Reverting")
        print(f"   Started: {current_trend['start'].date()}")
        print(f"   Duration: {current_trend['duration']} days")
        print(f"   Range: {current_trend['magnitude'] * 100:.1f} bp")
        print(f"   Strength: {current_trend['strength']:.2f}")
    else:
        trend_type = "Uptrend" if current_trend['direction'] == 1 else "Downtrend"
        print(f"\n📈 Current Trend: {trend_type}")
        print(f"   Started: {current_trend['start'].date()}")
        print(f"   Duration: {current_trend['duration']} days")
        print(f"   Magnitude: {current_trend['magnitude'] * 100:.1f} bp")
        print(f"   Strength: {current_trend['strength']:.2f}")

    # Forecast trend continuation probabilities
    continuation_probs = calculate_trend_probabilities(current_trend, trace, max_days=10)

    print("\n⏱️ Trend Continuation Probabilities:")
    for days, prob in continuation_probs.items():
        print(f"   Next {days} days: {prob:.2f}")

    # Generate seasonal context with weekend handling
    current_month = calendar.month_name[datetime.now().month]
    current_dow = calendar.day_name[datetime.now().weekday()]
    current_quarter = f"Q{(datetime.now().month - 1) // 3 + 1}"

    print("\n🗓️ Seasonal Context:")
    print(
        f"   Current Month: {current_month} (historical avg change: {seasonal_results['monthly'].loc[current_month, 'mean']:.2f} bp)")

    # Check if current day is a weekday before trying to access its seasonal data
    if current_dow in seasonal_results['day_of_week'].index:
        print(
            f"   Current Day: {current_dow} (historical avg change: {seasonal_results['day_of_week'].loc[current_dow, 'mean']:.2f} bp)")
    else:
        print(f"   Current Day: {current_dow} (No trading - weekend)")

    print(
        f"   Current Quarter: {current_quarter} (historical avg change: {seasonal_results['quarterly'].loc[current_quarter, 'mean']:.2f} bp)")

    # Display volatility regime
    current_volatility = prices.pct_change().iloc[-21:].std() * np.sqrt(252)
    if current_volatility > high_vol_threshold:
        vol_regime = "high"
    elif current_volatility < low_vol_threshold:
        vol_regime = "low"
    else:
        vol_regime = "medium"

    print(f"\n📉 Current Volatility Regime: {vol_regime.upper()}")
    print(f"   Annualized Volatility: {current_volatility * 100:.1f}%")
    print(f"   Historical Median: {median_vol * 100:.1f}%")

    # Create visualizations
    plt.figure(figsize=(12, 8))
    plot_trends_on_price(prices, trends)
    plt.tight_layout()

    plt.figure(figsize=(14, 10))
    plot_bayesian_results(trace, trends, current_trend, continuation_probs)

    plt.figure(figsize=(14, 7))
    plot_seasonality_analysis(seasonal_results, current_trend)

    # Plot backtest results
    plt.figure(figsize=(16, 12))
    plot_backtest_results(results, performance, trades)

    plt.show()


def prepare_seasonal_features(trends):
    """Extract seasonal features from trend data for Bayesian modeling"""
    # Initialize empty features dict
    features = {
        'month': [],
        'day_of_week': [],
        'quarter': []
    }

    # Extract features from each trend
    for trend in trends:
        start_date = trend['start']

        # Extract month (1-12)
        features['month'].append(start_date.month)

        # Extract day of week (0-4 for Mon-Fri)
        features['day_of_week'].append(start_date.dayofweek)

        # Extract quarter (1-4)
        features['quarter'].append(start_date.quarter)

    return features


def extract_trends(prices, min_trend_duration=2, coefficients=None, periods=None):
    """
    Enhanced trend detection specifically tuned for rates markets
    """
    # Calculate indicators - pass coefficients and periods as a tuple if available
    wavelet_data = None
    if coefficients is not None and periods is not None:
        wavelet_data = (coefficients, periods)

    indicators = calculate_indicators(prices, wavelet_data)

    # Use the more rates-sensitive trend change detection
    trends = identify_trend_changes(prices, indicators, wavelet_data, min_trend_duration=min_trend_duration)

    # Debug information
    print(f"\n📊 Detected {len(trends)} historical trends:")

    # Show summary of last few trends
    for i, trend in enumerate(trends[-5:], start=len(trends) - 4):
        print(f"Trend {i + 1}:")
        print(f"Start: {trend['start'].date()}")
        print(f"End: {trend['end'].date()}")
        print(f"Duration: {trend['duration']} days")
        print(f"Direction: {'Up' if trend['direction'] == 1 else 'Down'}")
        print(f"Strength: {trend['strength']:.2f}")
        print(f"Change: {trend['magnitude'] * 100:.1f} bp")
        print()

    return trends


def plot_bayesian_results(trace, trends, current_trend, continuation_probs):
    """Plot Bayesian model results with support for oscillating markets"""
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    # Determine if we're in an oscillating market or trending market
    is_oscillating = current_trend.get('type', '') == 'oscillating'

    # ==== Plot 1: Trend Duration Distributions ====
    ax1 = plt.subplot(gs[0, 0])

    # Get posterior samples
    up_duration = trace.posterior['up_duration_mu'].values.flatten()
    down_duration = trace.posterior['down_duration_mu'].values.flatten()
    osc_duration = trace.posterior['oscillation_duration_mu'].values.flatten()

    # Plot histograms
    ax1.hist(up_duration, bins=20, alpha=0.6, color='green', label='Uptrend Duration')
    ax1.hist(down_duration, bins=20, alpha=0.6, color='red', label='Downtrend Duration')
    ax1.hist(osc_duration, bins=20, alpha=0.6, color='purple', label='Oscillation Duration')

    # Add vertical lines for mean values
    ax1.axvline(np.mean(up_duration), color='green', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(down_duration), color='red', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(osc_duration), color='purple', linestyle='--', linewidth=2)

    # Highlight current trend duration
    ax1.axvline(current_trend['duration'], color='blue', linestyle='-', linewidth=2,
                label=f'Current: {current_trend["duration"]} days')

    ax1.set_title('Expected Trend Duration (Trading Days)', fontsize=14)
    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # ==== Plot 2: Continuation Probabilities ====
    ax2 = plt.subplot(gs[0, 1])

    # Plot trend continuation probabilities
    days = list(continuation_probs.keys())
    probs = list(continuation_probs.values())

    # Determine colors based on trend type
    if is_oscillating:
        bar_colors = ['purple'] * len(days)
        title = 'Probability of Oscillation Continuing'
    else:
        direction = current_trend['direction']
        bar_colors = ['green' if direction == 1 else 'red'] * len(days)
        title = f"Probability of {'Uptrend' if direction == 1 else 'Downtrend'} Continuing"

    # Plot bars
    bars = ax2.bar(days, probs, color=bar_colors, alpha=0.7)

    # Add value labels
    for i, prob in enumerate(probs):
        ax2.text(days[i], prob + 0.02, f'{prob:.2f}', ha='center')

    # Plot 50% reference line
    ax2.axhline(0.5, color='black', linestyle='--', alpha=0.7)

    ax2.set_xlabel('Additional Days', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title(title, fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # ==== Plot 3: Magnitude Distribution ====
    ax3 = plt.subplot(gs[1, 0])

    # Different magnitude metrics for different market types
    if is_oscillating:
        # For oscillating markets, plot the range
        range_samples = trace.posterior['oscillation_range_mu'].values.flatten() * 100  # Convert to bp
        ax3.hist(range_samples, bins=20, color='purple', alpha=0.7)
        ax3.axvline(np.mean(range_samples), color='purple', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(range_samples):.1f} bp')

        # Mark current range
        current_range = current_trend['magnitude'] * 100  # Convert to bp
        ax3.axvline(current_range, color='blue', linestyle='-', linewidth=2,
                    label=f'Current: {current_range:.1f} bp')

        ax3.set_title('Expected Oscillation Range (Basis Points)', fontsize=14)
    else:
        # For trending markets, plot the magnitude
        magnitude_samples = trace.posterior['magnitude_mu'].values.flatten() * 100  # Convert to bp
        ax3.hist(magnitude_samples, bins=20,
                 color='green' if current_trend['direction'] == 1 else 'red', alpha=0.7)

        ax3.axvline(np.mean(magnitude_samples),
                    color='green' if current_trend['direction'] == 1 else 'red',
                    linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(magnitude_samples):.1f} bp')

        # Mark current magnitude
        current_mag = current_trend['magnitude'] * 100  # Convert to bp
        ax3.axvline(current_mag, color='blue', linestyle='-', linewidth=2,
                    label=f'Current: {current_mag:.1f} bp')

        ax3.set_title('Expected Trend Magnitude (Basis Points)', fontsize=14)

    ax3.set_xlabel('Magnitude (bp)', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # ==== Plot 4: Regime Transition Analysis ====
    ax4 = plt.subplot(gs[1, 1])

    # Count transitions between regimes
    regime_transitions = count_regime_transitions(trends)

    # Calculate transition probabilities
    # We'll simplify to three states: Uptrend, Downtrend, Oscillating
    states = ['Uptrend', 'Downtrend', 'Oscillating']

    # Create transition matrix visualization
    transition_matrix = np.zeros((3, 3))

    # Fill transition matrix with available data
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            key = (from_state, to_state)
            if key in regime_transitions:
                transition_matrix[i, j] = regime_transitions[key]

    # Normalize rows to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_probabilities = transition_matrix / row_sums

    # Create heatmap
    im = ax4.imshow(transition_probabilities, cmap='YlOrRd', vmin=0, vmax=1)

    # Add text annotations
    for i in range(3):
        for j in range(3):
            text = ax4.text(j, i, f'{transition_probabilities[i, j]:.2f}',
                            ha="center", va="center",
                            color="black" if transition_probabilities[i, j] < 0.7 else "white",
                            fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('Transition Probability', fontsize=10)

    # Set ticks and labels
    ax4.set_xticks(np.arange(len(states)))
    ax4.set_yticks(np.arange(len(states)))
    ax4.set_xticklabels(states)
    ax4.set_yticklabels(states)
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Mark the current state
    if is_oscillating:
        current_idx = 2  # Oscillating
    else:
        current_idx = 0 if current_trend['direction'] == 1 else 1  # Uptrend or Downtrend

    # Add a rectangle around the current state
    ax4.add_patch(plt.Rectangle((current_idx - 0.5, -0.5), 1, 1, fill=False,
                                edgecolor='blue', lw=3, clip_on=False))

    ax4.set_title('Market Regime Transition Probabilities', fontsize=14)
    ax4.set_xlabel('To State', fontsize=12)
    ax4.set_ylabel('From State', fontsize=12)

    # Add explanation
    fig.text(0.5, 0.01,
             "Analysis shows expected duration and magnitude based on historical patterns.\nOscillating markets tend to have shorter duration with smaller changes.",
             ha='center', fontsize=10, style='italic')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return fig


def count_regime_transitions(trends):
    """Count transitions between market regimes"""
    transitions = {}

    for i in range(1, len(trends)):
        prev_trend = trends[i - 1]
        curr_trend = trends[i]

        # Determine previous state
        if prev_trend.get('type', '') == 'oscillating':
            prev_state = 'Oscillating'
        else:
            prev_state = 'Uptrend' if prev_trend['direction'] == 1 else 'Downtrend'

        # Determine current state
        if curr_trend.get('type', '') == 'oscillating':
            curr_state = 'Oscillating'
        else:
            curr_state = 'Uptrend' if curr_trend['direction'] == 1 else 'Downtrend'

        # Count transition
        key = (prev_state, curr_state)
        transitions[key] = transitions.get(key, 0) + 1

    return transitions


def plot_trends_on_price(prices, trends):
    """Plot price chart with trend periods highlighted, including oscillating regimes"""
    plt.figure(figsize=(14, 7))

    # Plot price data
    plt.plot(prices.index, prices, 'k-', alpha=0.7, label='Price')

    # Define colors for different trend types
    uptrend_color = 'green'
    downtrend_color = 'red'
    oscillating_color = 'purple'

    # Highlight trend periods with appropriate colors
    for trend in trends:
        if 'type' in trend and trend['type'] == 'oscillating':
            color = oscillating_color
            label = 'Oscillating'
            alpha = 0.3
            # For oscillating trends, add a pattern
            plt.axvspan(trend['start'], trend['end'], color=color, alpha=alpha, label=label)

            # Add oscillation markers - zigzag line
            dates = pd.date_range(trend['start'], trend['end'], periods=10)
            mid_price = (trend['start_price'] + trend['end_price']) / 2
            amplitude = (trend['magnitude'] * 50)  # Exaggerate for visibility
            oscillation = mid_price + amplitude * np.sin(np.linspace(0, 4 * np.pi, len(dates)))
            plt.plot(dates, oscillation, color=color, alpha=0.8, linewidth=1.5)

        else:
            # Regular up or down trend
            if trend['direction'] == 1:
                color = uptrend_color
                label = 'Uptrend'
            else:
                color = downtrend_color
                label = 'Downtrend'

            alpha = 0.4 if not trend.get('is_recent', False) else 0.6
            plt.axvspan(trend['start'], trend['end'], color=color, alpha=alpha, label=label)

    # Add annotations for the most recent trend
    if trends:
        current_trend = trends[-1]
        start_date = current_trend['start']
        end_date = current_trend['end']

        if 'type' in current_trend and current_trend['type'] == 'oscillating':
            trend_text = f"Oscillating Market\nDuration: {current_trend['duration']} days\nRange: {current_trend['magnitude'] * 100:.1f} bp"
        else:
            direction_text = "Uptrend" if current_trend['direction'] == 1 else "Downtrend"
            trend_text = f"{direction_text}\nDuration: {current_trend['duration']} days\nChange: {current_trend['magnitude'] * 100:.1f} bp"

        # Add text annotation for current trend
        plt.annotate(trend_text,
                     xy=(end_date, prices[end_date]),
                     xytext=(15, 0), textcoords='offset points',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                     fontsize=10, fontweight='bold')

        # Mark start and end points
        plt.scatter([start_date, end_date], [prices[start_date], prices[end_date]],
                    color='blue', s=50, zorder=5)

    # Clean up and add labels
    plt.title('Price Chart with Trend Analysis', fontsize=14)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=uptrend_color, alpha=0.6, label='Uptrend'),
        Patch(facecolor=downtrend_color, alpha=0.6, label='Downtrend'),
        Patch(facecolor=oscillating_color, alpha=0.6, label='Oscillating'),
        plt.Line2D([0], [0], color='k', alpha=0.7, label='Price')
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()


def plot_enhanced_probabilities(continuation_probs, trace, current_trend, expected_magnitude, magnitude_uncertainty,
                                seasonal_adjustments=None):
    """Create an enhanced dashboard of probability visualizations"""
    # Create a larger figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Enhanced Trend Analysis Dashboard - {"Uptrend" if current_trend["direction"] == 1 else "Downtrend"}',
                 fontsize=16, y=0.98)

    # Define grid for subplots
    gs = fig.add_gridspec(2, 3)

    # 1. Trend continuation vs time with confidence intervals
    ax1 = fig.add_subplot(gs[0, 0])
    days = list(continuation_probs.keys())
    probs = list(continuation_probs.values())

    # Calculate 80% confidence intervals (not statistically rigorous but visually helpful)
    lower_ci = [max(0.05, p - 0.15) for p in probs]
    upper_ci = [min(0.95, p + 0.15) for p in probs]

    # Plot with confidence intervals
    ax1.plot(days, probs, 'b-', linewidth=2, marker='o')
    ax1.fill_between(days, lower_ci, upper_ci, alpha=0.2, color='blue')

    # Add seasonal adjustments if available
    if seasonal_adjustments:
        # Plot the unadjusted probabilities as dashed line
        unadjusted_probs = [max(0.1, min(0.9, p - seasonal_adjustments.get(d, 0))) for p, d in zip(probs, days)]
        ax1.plot(days, unadjusted_probs, 'b--', linewidth=1, alpha=0.7, label='Before Seasonal Adjustment')

        # Add arrows to show the seasonal adjustment effect
        for i, (day, prob) in enumerate(zip(days, probs)):
            unadj_prob = unadjusted_probs[i]

            # Only show arrows where adjustment is significant
            if abs(prob - unadj_prob) > 0.03:
                arrow_props = dict(
                    arrowstyle='-|>',
                    color='purple' if prob > unadj_prob else 'orange',
                    alpha=0.7,
                    linewidth=1,
                    mutation_scale=10
                )

                ax1.annotate("", xy=(day, prob), xytext=(day, unadj_prob),
                             arrowprops=arrow_props)

    # Add horizontal lines for probability thresholds
    ax1.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Strong continuation')
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    ax1.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Likely reversal')

    ax1.set_xlabel('Days Ahead', fontsize=10)
    ax1.set_ylabel('Probability', fontsize=10)

    # Update title to indicate seasonal adjustment
    if seasonal_adjustments:
        ax1.set_title('Trend Continuation Probability vs Time\n(With Seasonal Adjustment)', fontsize=12)
    else:
        ax1.set_title('Trend Continuation Probability vs Time', fontsize=12)

    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)

    # 2. Magnitude vs probability with correlation effect
    ax2 = fig.add_subplot(gs[0, 1])

    # Get parameters from the trace
    if current_trend['direction'] == 1:
        duration_samples = trace.posterior['up_duration_mu'].values.flatten()
    else:
        duration_samples = trace.posterior['down_duration_mu'].values.flatten()

    duration_mu = np.mean(duration_samples)
    duration_sigma = np.std(duration_samples)

    # Extract correlation value if available
    has_correlation = 'duration_magnitude_correlation' in trace.posterior
    if has_correlation:
        correlation = np.mean(trace.posterior['duration_magnitude_correlation'].values)
    else:
        correlation = 0

    # Calculate current trend standardized duration
    current_duration = current_trend['duration']
    z_duration = (current_duration - duration_mu) / (duration_sigma + 1e-6)

    # Generate range of potential magnitude changes (as percentage of current magnitude)
    pct_magnitudes = np.linspace(0.1, 2.0, 30)  # More granular scale
    magnitudes = current_trend['magnitude'] * pct_magnitudes

    # Calculate probabilities for each magnitude with correlation effect
    magnitude_probs = []
    for mag in magnitudes:
        # Baseline probability based on current duration
        base_prob = 1 - stats.norm.cdf(z_duration)

        # Adjust for magnitude
        mag_z_score = (mag - expected_magnitude) / (magnitude_uncertainty + 1e-6)
        mag_prob = stats.norm.pdf(mag_z_score) / stats.norm.pdf(0)

        # Apply correlation effect if available
        if has_correlation and abs(correlation) > 0.1:
            # Larger magnitudes with positive correlation should reinforce continuation
            corr_effect = correlation * mag_z_score * 0.1
            joint_prob = base_prob * (1 + corr_effect) * mag_prob
        else:
            joint_prob = base_prob * mag_prob

        magnitude_probs.append(max(0.1, min(0.9, joint_prob)))

    # Plot the probability curve with confidence bands
    ax2.plot(pct_magnitudes, magnitude_probs, 'r-', linewidth=2)

    # Calculate confidence intervals
    mag_lower_ci = [max(0.05, p - 0.15) for p in magnitude_probs]
    mag_upper_ci = [min(0.95, p + 0.15) for p in magnitude_probs]

    ax2.fill_between(pct_magnitudes, mag_lower_ci, mag_upper_ci, alpha=0.2, color='red')

    # Mark current magnitude
    ax2.axvline(x=1.0, color='black', linestyle='--', alpha=0.7,
                label=f'Current ({current_trend["magnitude"]:.4f})')

    # Add optimal magnitude if there's a peak
    peak_idx = np.argmax(magnitude_probs)
    ax2.axvline(x=pct_magnitudes[peak_idx], color='green', linestyle='-', alpha=0.5,
                label=f'Optimal ({pct_magnitudes[peak_idx]:.2f}×)')

    # Add probability thresholds
    ax2.axhline(0.7, color='green', linestyle=':', alpha=0.5)
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(0.3, color='red', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Magnitude (× Current)', fontsize=10)
    ax2.set_ylabel('Probability', fontsize=10)
    ax2.set_title('Trend Continuation Probability vs Magnitude', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)

    # 3. Volatility-Specific Probability Analysis
    ax3 = fig.add_subplot(gs[0, 2])

    # Check if we have volatility regime specific parameters
    has_vol_regime = 'up_duration_mu_low_vol' in trace.posterior

    if has_vol_regime:
        # Get regime boundary
        regime_boundary = np.mean(trace.posterior['regime_boundary'].values)

        # Create two sets of probabilities - high vol and low vol
        days_high_vol = days.copy()
        days_low_vol = days.copy()

        # Use appropriate parameters for each regime
        if current_trend['direction'] == 1:
            dur_mu_high = np.mean(trace.posterior['up_duration_mu_high_vol'].values)
            dur_sigma_high = np.std(trace.posterior['up_duration_mu_high_vol'].values)

            dur_mu_low = np.mean(trace.posterior['up_duration_mu_low_vol'].values)
            dur_sigma_low = np.std(trace.posterior['up_duration_mu_low_vol'].values)
        else:
            dur_mu_high = np.mean(trace.posterior['down_duration_mu_high_vol'].values)
            dur_sigma_high = np.std(trace.posterior['down_duration_mu_high_vol'].values)

            dur_mu_low = np.mean(trace.posterior['down_duration_mu_low_vol'].values)
            dur_sigma_low = np.std(trace.posterior['down_duration_mu_low_vol'].values)

        # Calculate probabilities for high and low volatility regimes
        probs_high_vol = []
        probs_low_vol = []

        for day in days:
            z_high = (current_duration + day - dur_mu_high) / (dur_sigma_high + 1e-6)
            prob_high = 1 - stats.norm.cdf(z_high)
            probs_high_vol.append(max(0.1, min(0.9, prob_high)))

            z_low = (current_duration + day - dur_mu_low) / (dur_sigma_low + 1e-6)
            prob_low = 1 - stats.norm.cdf(z_low)
            probs_low_vol.append(max(0.1, min(0.9, prob_low)))

        # Plot high and low volatility probabilities
        ax3.plot(days_high_vol, probs_high_vol, 'r-', linewidth=2, marker='o',
                 label='High Volatility Regime')
        ax3.plot(days_low_vol, probs_low_vol, 'g-', linewidth=2, marker='s',
                 label='Low Volatility Regime')

        # Highlight the current regime
        current_vol = current_trend.get('volatility', 0.01)
        regime_text = "High Volatility" if current_vol > regime_boundary else "Low Volatility"

        ax3.set_title(f'Volatility Regime Analysis\nCurrent: {regime_text}', fontsize=12)
    else:
        # If we don't have volatility regimes, show a message
        ax3.text(0.5, 0.5, "Volatility Regime Analysis\nNot Available",
                 ha='center', va='center', fontsize=12)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title('Volatility Regime Analysis', fontsize=12)

    ax3.set_xlabel('Days Ahead', fontsize=10)
    ax3.set_ylabel('Probability', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    if has_vol_regime:
        ax3.legend(fontsize=8)

    # 4. Duration-Magnitude Heatmap (bottom left) - IMPROVED VERSION
    ax4 = fig.add_subplot(gs[1, 0])

    # Create a more detailed heatmap with better ranges
    max_duration = 40
    magnitude_steps = 20

    # Create grid of durations and magnitudes
    durations = np.arange(1, max_duration + 1)

    # Calculate magnitude range based on historical data
    base_magnitude = current_trend['magnitude']
    magnitude_range = np.linspace(base_magnitude * 0.5, base_magnitude * 2, magnitude_steps)
    probabilities = np.zeros((len(magnitude_range), len(durations)))

    # Get correlation for joint distribution
    if has_correlation:
        correlation_effect = correlation
    else:
        correlation_effect = 0

    # Calculate joint probabilities for each combination with correlation
    for i, mag in enumerate(magnitude_range):
        for j, dur in enumerate(durations):
            # Get individual probabilities for duration and magnitude
            if current_duration < dur:
                # If we're looking at future durations
                # Probability that trend continues to this duration
                z_score_dur = (dur - current_duration - duration_mu) / (duration_sigma + 1e-6)
                p_dur = 1 - stats.norm.cdf(z_score_dur)
            else:
                # If we're looking at past durations (almost certainly happened)
                p_dur = 0.95

            # Probability of seeing this magnitude
            mag_ratio = mag / base_magnitude

            # Adjust based on trend direction for better intuition
            if current_trend['direction'] == 1:  # Uptrend
                # For uptrends, higher magnitudes are considered "stronger"
                if mag_ratio > 1.0:
                    # Higher magnitude is less likely the higher it goes
                    p_mag = np.exp(-0.5 * ((mag_ratio - 1.0) / 0.5) ** 2)
                else:
                    # Lower magnitude is very likely
                    p_mag = 1.0 - 0.5 * (1.0 - mag_ratio) ** 2
            else:  # Downtrend
                # For downtrends, lower magnitudes are considered "stronger"
                if mag_ratio < 1.0:
                    # Lower magnitude is less likely the lower it goes
                    p_mag = np.exp(-0.5 * ((1.0 - mag_ratio) / 0.5) ** 2)
                else:
                    # Higher magnitude is very likely
                    p_mag = 1.0 - 0.5 * (mag_ratio - 1.0) ** 2

            # Combine probabilities, accounting for correlation
            if abs(correlation_effect) < 0.01:
                # No correlation: independent probabilities
                joint_prob = p_dur * p_mag
            else:
                # With correlation: use a weighted combination
                weight = 0.5 + 0.5 * abs(correlation_effect)
                if correlation_effect > 0:
                    # Positive correlation: reinforce when both are high/low
                    joint_prob = weight * min(p_dur, p_mag) + (1 - weight) * (p_dur * p_mag)
                else:
                    # Negative correlation: reinforce when one is high and one is low
                    joint_prob = weight * (p_dur * (1 - p_mag) + p_mag * (1 - p_dur)) / 2 + (1 - weight) * (
                                p_dur * p_mag)

            # Ensure probability is between 0.1 and 0.9
            probabilities[i, j] = max(0.1, min(0.9, joint_prob))

    # Create a more intuitive heatmap
    im = ax4.imshow(probabilities,
                    aspect='auto',
                    origin='lower',
                    extent=[0, max_duration, magnitude_range[0], magnitude_range[-1]],
                    cmap='viridis')  # Changed colormap for better visualization

    # Add colorbar
    plt.colorbar(im, ax=ax4, label='Joint Probability')

    # Mark current trend
    ax4.axvline(x=current_trend['duration'],
                color='white',
                linestyle='--',
                alpha=0.7)
    ax4.axhline(y=current_trend['magnitude'],
                color='white',
                linestyle=':',
                alpha=0.7)

    # Mark point of current duration and magnitude
    ax4.plot(current_trend['duration'], current_trend['magnitude'], 'wo',
             markersize=8, markeredgecolor='black', label='Current State')

    # Calculate and mark most likely future state
    if current_duration < max_duration - 5:
        # Find the highest probability future state
        # Only look at plausible future states (5+ days ahead)
        future_slice = probabilities[:, current_duration + 5:].copy()
        max_idx = np.unravel_index(np.argmax(future_slice), future_slice.shape)
        future_mag_idx, future_dur_idx = max_idx
        future_mag = magnitude_range[future_mag_idx]
        future_dur = durations[future_dur_idx + current_duration + 5]

        # Mark this point
        ax4.plot(future_dur, future_mag, 'yo',
                 markersize=8, markeredgecolor='black', label='Most Likely Future')

        # Draw an arrow from current to future state
        ax4.annotate("",
                     xy=(future_dur, future_mag),
                     xytext=(current_trend['duration'], current_trend['magnitude']),
                     arrowprops=dict(facecolor='yellow', shrink=0.05, width=2, alpha=0.7))

    # Improve title and labels
    ax4.set_title(
        f'Duration-Magnitude Probability Map\n{"Uptrend" if current_trend["direction"] == 1 else "Downtrend"}',
        fontsize=12)
    ax4.set_xlabel('Duration (days)', fontsize=10)
    ax4.set_ylabel('Magnitude', fontsize=10)
    ax4.legend(fontsize=8, loc='upper right')

    # 5. Trend Strength Effect (bottom middle)
    ax5 = fig.add_subplot(gs[1, 1])

    # Check if we have strength parameter
    has_strength = 'duration_magnitude_correlation' in trace.posterior

    if has_strength:
        # Get strength parameters
        strength_mu = np.mean(trace.posterior['duration_magnitude_correlation'].values)

        # Generate different strength scenarios
        strength_values = [0.3, 0.5, 0.7, 0.9]  # weak to strong

        # For each strength value, calculate continuation probability
        for strength in strength_values:
            # Calculate strength-adjusted probabilities
            strength_probs = []

            for day in days:
                # Base probability from duration
                z_score = (current_duration + day - duration_mu) / (duration_sigma + 1e-6)
                base_prob = 1 - stats.norm.cdf(z_score)

                # Adjust by strength (strong trends persist longer)
                strength_effect = (strength - 0.5) * 0.4  # -0.2 to +0.2 adjustment
                adj_prob = base_prob * (1 + strength_effect)

                strength_probs.append(max(0.1, min(0.9, adj_prob)))

            # Plot this strength scenario
            ax5.plot(days, strength_probs, '-', linewidth=2, marker='.',
                     label=f'Strength = {strength:.1f}')

        # Highlight current trend strength
        current_strength = current_trend.get('strength', 0.5)
        ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        ax5.set_title(f'Effect of Trend Strength on Continuation\nCurrent Strength: {current_strength:.2f}',
                      fontsize=12)
    else:
        # If strength effect not modeled
        ax5.text(0.5, 0.5, "Trend Strength Analysis\nNot Available",
                 ha='center', va='center', fontsize=12)
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.set_title('Trend Strength Analysis', fontsize=12)

    ax5.set_xlabel('Days Ahead', fontsize=10)
    ax5.set_ylabel('Probability', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)
    if has_strength:
        ax5.legend(fontsize=8)

    # 6. Trading Strategy Implications (bottom right)
    ax6 = fig.add_subplot(gs[1, 2])

    # Generate trading strategy implications based on probabilities
    # We'll use a simple scoring system

    # Calculate average probability over next 5 days
    short_term_prob = np.mean([continuation_probs[d] for d in [1, 2, 3, 4, 5] if d in continuation_probs])

    # Calculate medium-term probability (next 10-15 days)
    medium_term_days = [d for d in [10, 11, 12, 13, 14, 15] if d in continuation_probs]
    medium_term_prob = np.mean([continuation_probs[d] for d in medium_term_days]) if medium_term_days else 0.5

    # Calculate long-term probability (20+ days)
    long_term_days = [d for d in [20, 25, 30] if d in continuation_probs]
    long_term_prob = np.mean([continuation_probs[d] for d in long_term_days]) if long_term_days else 0.5

    # Calculate optimal holding period (where probability drops below 0.5)
    holding_period = 0
    for day in sorted(continuation_probs.keys()):
        if continuation_probs[day] < 0.5:
            holding_period = day - 1
            break
    if holding_period == 0 and continuation_probs:  # If never drops below 0.5
        holding_period = max(continuation_probs.keys())

    # Calculate expected magnitude at optimal holding
    if 'duration_magnitude_correlation' in trace.posterior:
        corr = np.mean(trace.posterior['duration_magnitude_correlation'].values)
        holding_effect = (holding_period / current_duration - 1) * corr
        expected_peak_mag = expected_magnitude * (1 + holding_effect)
    else:
        expected_peak_mag = expected_magnitude

    # Calculate position sizing recommendation (0-100%)
    trend_confidence = short_term_prob * 0.4 + medium_term_prob * 0.4 + long_term_prob * 0.2
    vol_adjustment = 1.0
    if has_vol_regime:
        current_vol = current_trend.get('volatility', 0.01)
        regime_boundary_val = np.mean(trace.posterior['regime_boundary'].values)
        vol_adjustment = 0.7 if current_vol > regime_boundary_val else 1.0

    position_size = int(trend_confidence * 100 * vol_adjustment)

    # Generate strategy text
    direction = "LONG" if current_trend['direction'] == 1 else "SHORT"
    strategy_text = [
        f"Trading Strategy Implications\n",
        f"Direction: {direction}",
        f"Confidence: {trend_confidence:.2f}",
        f"Optimal Holding Period: {holding_period} days",
        f"Expected Magnitude: {expected_peak_mag:.4f}",
        f"Position Size: {position_size}%",
        f"\nStrength:",
        f"Short-term (1-5d): {'Strong' if short_term_prob > 0.65 else 'Moderate' if short_term_prob > 0.5 else 'Weak'}",
        f"Medium-term (10-15d): {'Strong' if medium_term_prob > 0.65 else 'Moderate' if medium_term_prob > 0.5 else 'Weak'}",
        f"Long-term (20d+): {'Strong' if long_term_prob > 0.65 else 'Moderate' if long_term_prob > 0.5 else 'Weak'}"
    ]

    # Add seasonal factors if available
    if seasonal_adjustments:
        # Calculate average seasonal effect
        avg_seasonal = np.mean(list(seasonal_adjustments.values()))
        effect = "positive" if avg_seasonal > 0 else "negative"
        seasonal_magnitude = "strong" if abs(avg_seasonal) > 0.1 else "moderate" if abs(avg_seasonal) > 0.05 else "weak"

        strategy_text.extend([
            f"\nSeasonal Factors:",
            f"Current Effect: {seasonal_magnitude} {effect}",
            f"Adjustment: {avg_seasonal:.3f}"
        ])

    # Display strategy recommendations
    ax6.text(0.5, 0.5, "\n".join(strategy_text),
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.2))
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.set_title('Trading Strategy Implications', fontsize=12)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle


def plot_bayesian_seasonal_effects(trace):
    """Plot multiple Bayesian model parameters with their distributions"""
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Bayesian Model Parameter Distributions', fontsize=16)

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # 1. Plot Duration-Magnitude Correlation
    ax1 = axes[0]
    correlation_samples = trace.posterior['duration_magnitude_correlation'].values.flatten()
    correlation_mean = np.mean(correlation_samples)
    correlation_hdi = np.percentile(correlation_samples, [2.5, 97.5])

    # Plot histogram of correlation samples
    ax1.hist(correlation_samples, bins=30, alpha=0.7, color='skyblue')
    ax1.axvline(correlation_mean, color='red', linestyle='-', linewidth=2, label=f'Mean: {correlation_mean:.3f}')
    ax1.axvline(correlation_hdi[0], color='black', linestyle='--', linewidth=1.5, label=f'95% HDI')
    ax1.axvline(correlation_hdi[1], color='black', linestyle='--', linewidth=1.5)
    ax1.axvline(0, color='gray', linestyle='-', alpha=0.5)

    ax1.set_title('Duration-Magnitude Correlation', fontsize=12)
    ax1.set_xlabel('Correlation Value', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.legend(loc='best', fontsize=8)

    # 2. Plot Uptrend Duration Mean
    ax2 = axes[1]
    up_dur_samples = trace.posterior['up_duration_mu'].values.flatten()
    up_dur_mean = np.mean(up_dur_samples)
    up_dur_hdi = np.percentile(up_dur_samples, [2.5, 97.5])

    ax2.hist(up_dur_samples, bins=30, alpha=0.7, color='green')
    ax2.axvline(up_dur_mean, color='red', linestyle='-', linewidth=2, label=f'Mean: {up_dur_mean:.1f}d')
    ax2.axvline(up_dur_hdi[0], color='black', linestyle='--', linewidth=1.5, label=f'95% HDI')
    ax2.axvline(up_dur_hdi[1], color='black', linestyle='--', linewidth=1.5)

    ax2.set_title('Uptrend Duration Mean', fontsize=12)
    ax2.set_xlabel('Days', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.legend(loc='best', fontsize=8)

    # 3. Plot Downtrend Duration Mean
    ax3 = axes[2]
    down_dur_samples = trace.posterior['down_duration_mu'].values.flatten()
    down_dur_mean = np.mean(down_dur_samples)
    down_dur_hdi = np.percentile(down_dur_samples, [2.5, 97.5])

    ax3.hist(down_dur_samples, bins=30, alpha=0.7, color='red')
    ax3.axvline(down_dur_mean, color='blue', linestyle='-', linewidth=2, label=f'Mean: {down_dur_mean:.1f}d')
    ax3.axvline(down_dur_hdi[0], color='black', linestyle='--', linewidth=1.5, label=f'95% HDI')
    ax3.axvline(down_dur_hdi[1], color='black', linestyle='--', linewidth=1.5)

    ax3.set_title('Downtrend Duration Mean', fontsize=12)
    ax3.set_xlabel('Days', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.legend(loc='best', fontsize=8)

    # 4. Plot Magnitude Mean
    ax4 = axes[3]
    mag_samples = trace.posterior['magnitude_mu'].values.flatten()
    mag_sigma_samples = trace.posterior['magnitude_sigma'].values.flatten()
    mag_mean = np.mean(mag_samples)
    mag_hdi = np.percentile(mag_samples, [2.5, 97.5])

    ax4.hist(mag_samples, bins=30, alpha=0.7, color='purple')
    ax4.axvline(mag_mean, color='orange', linestyle='-', linewidth=2, label=f'Mean: {mag_mean:.3f}')
    ax4.axvline(mag_hdi[0], color='black', linestyle='--', linewidth=1.5, label=f'95% HDI')
    ax4.axvline(mag_hdi[1], color='black', linestyle='--', linewidth=1.5)

    ax4.set_title('Magnitude Mean', fontsize=12)
    ax4.set_xlabel('Magnitude', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.legend(loc='best', fontsize=8)

    # Add interpretation
    fig.text(0.5, 0.01,
             "These distributions show the uncertainty in our model parameters.\nWider distributions indicate more uncertainty.",
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_volatility_regime_analysis(trends, trace):
    """Plot analysis of trend behavior in different volatility regimes"""
    # Separate trends by volatility regime
    high_vol_trends = [t for t in trends if t.get('vol_regime') == 'high']
    med_vol_trends = [t for t in trends if t.get('vol_regime') == 'medium']
    low_vol_trends = [t for t in trends if t.get('vol_regime') == 'low']

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Volatility Regime Analysis', fontsize=16)

    # 1. Duration by Volatility Regime
    ax1 = axes[0]

    # Calculate mean duration for each regime and direction
    regimes = ['low', 'medium', 'high']
    regime_labels = ['Low Vol', 'Medium Vol', 'High Vol']
    up_durations = []
    down_durations = []

    for regime_trends in [low_vol_trends, med_vol_trends, high_vol_trends]:
        if regime_trends:
            up_dur = np.mean([t['duration'] for t in regime_trends if t['direction'] == 1]) if any(
                t['direction'] == 1 for t in regime_trends) else 0
            down_dur = np.mean([t['duration'] for t in regime_trends if t['direction'] == -1]) if any(
                t['direction'] == -1 for t in regime_trends) else 0
            up_durations.append(up_dur)
            down_durations.append(down_dur)
        else:
            up_durations.append(0)
            down_durations.append(0)

    x = np.arange(len(regimes))
    width = 0.35

    ax1.bar(x - width / 2, up_durations, width, label='Uptrends', color='green', alpha=0.7)
    ax1.bar(x + width / 2, down_durations, width, label='Downtrends', color='red', alpha=0.7)

    # Add value labels
    for i, v in enumerate(up_durations):
        if v > 0:
            ax1.text(i - width / 2, v + 0.5, f'{v:.1f}d', ha='center', va='bottom', fontsize=9)

    for i, v in enumerate(down_durations):
        if v > 0:
            ax1.text(i + width / 2, v + 0.5, f'{v:.1f}d', ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(regime_labels)
    ax1.legend(loc='best')
    ax1.set_ylabel('Average Duration (Days)')
    ax1.set_title('Trend Duration by Volatility Regime')
    ax1.grid(axis='y', alpha=0.3)

    # 2. Magnitude by Volatility Regime
    ax2 = axes[1]

    # Calculate mean magnitude for each regime
    up_magnitudes = []
    down_magnitudes = []

    for regime_trends in [low_vol_trends, med_vol_trends, high_vol_trends]:
        if regime_trends:
            up_mag = np.mean([t['magnitude'] for t in regime_trends if t['direction'] == 1]) if any(
                t['direction'] == 1 for t in regime_trends) else 0
            down_mag = np.mean([t['magnitude'] for t in regime_trends if t['direction'] == -1]) if any(
                t['direction'] == -1 for t in regime_trends) else 0
            up_magnitudes.append(up_mag)
            down_magnitudes.append(down_mag)
        else:
            up_magnitudes.append(0)
            down_magnitudes.append(0)

    ax2.bar(x - width / 2, up_magnitudes, width, label='Uptrends', color='green', alpha=0.7)
    ax2.bar(x + width / 2, down_magnitudes, width, label='Downtrends', color='red', alpha=0.7)

    # Add value labels
    for i, v in enumerate(up_magnitudes):
        if v > 0:
            ax2.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    for i, v in enumerate(down_magnitudes):
        if v > 0:
            ax2.text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(regime_labels)
    ax2.legend(loc='best')
    ax2.set_ylabel('Average Magnitude')
    ax2.set_title('Trend Magnitude by Volatility Regime')
    ax2.grid(axis='y', alpha=0.3)

    # Add interpretation text
    fig.text(0.5, 0.01,
             "Higher volatility regimes typically have shorter trends with larger magnitude changes.",
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def analyze_seasonality(prices):
    """
    Analyze seasonal patterns in rates data using absolute changes (basis points)
    rather than percentage changes
    """
    # Create a copy of the data
    df = prices.copy().to_frame(name='price')

    # Calculate absolute changes in basis points (1 bp = 0.0001 for 10Y yield)
    df['change'] = df['price'].diff() * 100  # Convert to basis points

    # Extract date features
    df['date'] = df.index
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['week_of_year'] = df.index.isocalendar().week
    df['day_of_month'] = df.index.day

    # Ensure we have a complete month name
    df['month_name'] = df['date'].dt.strftime('%B')

    # Get day of week name
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_name'] = df['day_of_week'].apply(lambda x: days[x])

    # Create quarter name
    df['quarter_name'] = 'Q' + df['quarter'].astype(str)

    # Remove NaN values from the first row due to diff()
    df = df.dropna()

    # Calculate seasonal effects by day of week
    dow_effect = df.groupby('day_name')['change'].agg(['mean', 'std', 'count']).reset_index()
    dow_effect = dow_effect.sort_values(by='mean', ascending=False)  # Sort by effect strength

    # Calculate seasonal effects by month
    monthly_effect = df.groupby('month_name')['change'].agg(['mean', 'std', 'count']).reset_index()

    # Sort months in calendar order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_effect['month_name'] = pd.Categorical(monthly_effect['month_name'], categories=month_order, ordered=True)
    monthly_effect = monthly_effect.sort_values('month_name')

    # Calculate seasonal effects by quarter
    quarterly_effect = df.groupby('quarter_name')['change'].agg(['mean', 'std', 'count']).reset_index()
    quarterly_effect = quarterly_effect.sort_values('quarter_name')

    # Setting indices for easier lookup
    dow_effect = dow_effect.set_index('day_name')
    monthly_effect = monthly_effect.set_index('month_name')
    quarterly_effect = quarterly_effect.set_index('quarter_name')

    # Return as dictionary of results
    return {
        'day_of_week': dow_effect,
        'monthly': monthly_effect,
        'quarterly': quarterly_effect,
        'unit': 'basis points'  # Explicitly note we're using basis points now
    }


def plot_seasonality_analysis(seasonal_results, current_trend=None):
    """Plot seasonality analysis with basis point changes"""
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Seasonal Analysis - Basis Points Change', fontsize=16)

    # Plot monthly seasonality
    monthly_data = seasonal_results['monthly']
    ax1 = axes[0]
    months = monthly_data.index
    means = monthly_data['mean']
    stds = monthly_data['std']

    bars = ax1.bar(months, means, yerr=stds, alpha=0.7, capsize=5,
                   color=['green' if x > 0 else 'red' for x in means])

    # Highlight current month if provided
    if current_trend:
        current_month = calendar.month_name[current_trend['end'].month]
        if current_month in months:
            idx = months.get_loc(current_month)
            bars[idx].set_color('blue')
            bars[idx].set_alpha(1.0)
            ax1.text(idx, means[current_month] + 0.01, '← Current',
                     fontweight='bold', ha='right', va='bottom')

    ax1.set_title('Monthly Seasonality', fontsize=14)
    ax1.set_ylabel('Avg Change (Basis Points)', fontsize=12)
    ax1.set_xticklabels(months, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)

    # Plot day of week seasonality
    dow_data = seasonal_results['day_of_week']
    ax2 = axes[1]
    days = dow_data.index
    means = dow_data['mean']
    stds = dow_data['std']

    bars = ax2.bar(days, means, yerr=stds, alpha=0.7, capsize=5,
                   color=['green' if x > 0 else 'red' for x in means])

    # Highlight current day if provided and it's a weekday
    if current_trend:
        current_day = calendar.day_name[current_trend['end'].weekday()]
        if current_day in days:
            idx = days.get_loc(current_day)
            bars[idx].set_color('blue')
            bars[idx].set_alpha(1.0)
            ax2.text(idx, means[current_day] + 0.01, '← Current',
                     fontweight='bold', ha='right', va='bottom')
        elif current_day in ['Saturday', 'Sunday']:
            # Add a note for weekend days
            ax2.annotate(f'Current: {current_day} (Weekend)',
                         xy=(0.5, 0.95), xycoords='axes fraction',
                         ha='center', va='top',
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

    ax2.set_title('Day of Week Seasonality', fontsize=14)
    ax2.set_ylabel('Avg Change (Basis Points)', fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Plot quarterly seasonality
    quarterly_data = seasonal_results['quarterly']
    ax3 = axes[2]
    quarters = quarterly_data.index
    means = quarterly_data['mean']
    stds = quarterly_data['std']

    bars = ax3.bar(quarters, means, yerr=stds, alpha=0.7, capsize=5,
                   color=['green' if x > 0 else 'red' for x in means])

    # Highlight current quarter if provided
    if current_trend:
        current_quarter = f'Q{current_trend["end"].quarter}'
        if current_quarter in quarters:
            idx = quarters.get_loc(current_quarter)
            bars[idx].set_color('blue')
            bars[idx].set_alpha(1.0)
            ax3.text(idx, means[current_quarter] + 0.01, '← Current',
                     fontweight='bold', ha='right', va='bottom')

    ax3.set_title('Quarterly Seasonality', fontsize=14)
    ax3.set_ylabel('Avg Change (Basis Points)', fontsize=12)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)

    # Add a note about basis points
    fig.text(0.5, 0.01,
             "Note: All values represent average daily change in basis points (1/100 of 1%)",
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


if __name__ == "__main__":
    main()
