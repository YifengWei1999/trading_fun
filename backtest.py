import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Tuple


class Strategy(ABC):
    """Abstract base class for trading strategies"""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market data

        Parameters:
        -----------
        data : pd.DataFrame
            Market data with OHLCV and any additional columns

        Returns:
        --------
        pd.DataFrame
            Data with added signal column (1 for long, -1 for short, 0 for flat)
        """
        pass


class MovingAverageCrossover(Strategy):
    """Simple moving average crossover strategy"""

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate moving averages
        df['fast_ma'] = df['Close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['Close'].rolling(window=self.slow_period).mean()

        # Generate signals
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1

        # Generate positions (signal changes)
        df['position'] = df['signal'].shift(1)
        df['position'].fillna(0, inplace=True)

        return df


class VolumeSpikeMomentum(Strategy):
    """Strategy that follows price direction after volume spikes"""

    def __init__(self, volume_threshold: float = 2.0, lookback_period: int = 20):
        """
        Initialize the volume spike momentum strategy

        Parameters:
        -----------
        volume_threshold : float
            Multiple of average volume that constitutes a spike (e.g., 2.0 = 2x avg volume)
        lookback_period : int
            Period used to calculate the average volume
        """
        self.volume_threshold = volume_threshold
        self.lookback_period = lookback_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate rolling average volume
        df['avg_volume'] = df['Volume'].rolling(window=self.lookback_period).mean()

        # Identify volume spikes
        df['volume_spike'] = (df['Volume'] > (df['avg_volume'] * self.volume_threshold)).astype(int)

        # Price direction (1 for up, -1 for down, 0 for no change)
        df['price_direction'] = np.sign(df['Close'].diff())

        # Signal follows the direction 1 period after a volume spike
        df['signal'] = 0

        # Shift the volume spike to align with the next period
        spike_indices = df[df['volume_spike'] == 1].index

        # For each spike, get the direction of the next period and set as signal
        for idx in spike_indices:
            try:
                next_idx = df.index[df.index.get_loc(idx) + 1]
                if next_idx in df.index:
                    df.loc[next_idx, 'signal'] = df.loc[next_idx, 'price_direction']
            except (IndexError, KeyError):
                # Handle the case where the spike is at the last data point
                continue

        # Generate positions from signals
        df['position'] = df['signal'].shift(1)
        df['position'].fillna(0, inplace=True)

        return df


class Asset:
    """Class representing a tradable asset"""

    def __init__(self, symbol: str, asset_type: str = 'futures',
                 multiplier: float = 1.0, margin_requirement: float = 0.1):
        """
        Initialize an asset

        Parameters:
        -----------
        symbol : str
            Ticker symbol
        asset_type : str
            Type of asset (futures, stock, etc.)
        multiplier : float
            Contract multiplier for futures
        margin_requirement : float
            Initial margin requirement as a fraction of contract value
        """
        self.symbol = symbol
        self.asset_type = asset_type
        self.multiplier = multiplier
        self.margin_requirement = margin_requirement
        self.data = None

    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch market data for the asset"""
        print(f"Fetching {self.symbol} data ({start_date} to {end_date})...")
        df = yf.download(self.symbol, start=start_date, end=end_date, progress=False)

        if df.empty:
            raise ValueError(f"Received empty dataset for {self.symbol}")

        print(f"Data retrieved: {len(df)} trading days")
        self.data = df
        return df

    def set_custom_data(self, data: pd.DataFrame) -> None:
        """
        Set custom market data for the asset

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with OHLCV data, should have columns: Open, High, Low, Close, Volume
            and a DatetimeIndex
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Data is missing required columns: {missing_columns}")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex")

        self.data = data
        print(f"Custom data set for {self.symbol}: {len(data)} intervals")


class Portfolio:
    """Class for tracking portfolio performance"""

    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize portfolio

        Parameters:
        -----------
        initial_capital : float
            Initial capital for the portfolio
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> quantity
        self.trades = []
        self.equity_curve = []

    def calculate_returns(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio returns based on positions

        Parameters:
        -----------
        positions_df : pd.DataFrame
            DataFrame with positions and market data

        Returns:
        --------
        pd.DataFrame
            DataFrame with portfolio returns and metrics
        """
        df = positions_df.copy()

        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        df['strategy_returns'] = df['position'] * df['returns']

        # Calculate equity curve
        df['equity_curve'] = (1 + df['strategy_returns']).cumprod() * self.initial_capital

        # Calculate drawdowns
        df['peak'] = df['equity_curve'].cummax()
        df['drawdown'] = (df['equity_curve'] - df['peak']) / df['peak']

        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        df['strategy_cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1

        return df


class BacktestEngine:
    """Main backtesting engine"""

    def __init__(self, assets: List[Asset], strategy: Strategy,
                 start_date: str, end_date: str, initial_capital: float = 100000.0):
        """
        Initialize backtesting engine

        Parameters:
        -----------
        assets : List[Asset]
            List of assets to trade
        strategy : Strategy
            Trading strategy to use
        start_date : str
            Start date for backtest in 'YYYY-MM-DD' format
        end_date : str
            End date for backtest in 'YYYY-MM-DD' format
        initial_capital : float
            Initial capital for the backtest
        """
        self.assets = assets
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.portfolio = Portfolio(initial_capital)
        self.results = {}

    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Run the backtest

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of results for each asset
        """
        for asset in self.assets:
            # Fetch data
            data = asset.fetch_data(self.start_date, self.end_date)

            # Generate signals
            signals_df = self.strategy.generate_signals(data)

            # Calculate portfolio performance
            results_df = self.portfolio.calculate_returns(signals_df)

            self.results[asset.symbol] = results_df

        return self.results

    def calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for each asset

        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary of metrics for each asset
        """
        metrics = {}

        for symbol, df in self.results.items():
            # Calculate performance metrics
            total_return = df['strategy_cumulative_returns'].iloc[-1]
            annual_return = (1 + total_return) ** (252 / len(df)) - 1

            # Risk metrics
            daily_std = df['strategy_returns'].std()
            annualized_vol = daily_std * np.sqrt(252)
            sharpe_ratio = annual_return / annualized_vol if annualized_vol != 0 else 0
            max_drawdown = df['drawdown'].min()

            # Trade metrics
            df['trade'] = df['position'].diff().fillna(0)
            total_trades = (df['trade'] != 0).sum()

            metrics[symbol] = {
                'Total Return': total_return,
                'Annual Return': annual_return,
                'Annualized Volatility': annualized_vol,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Total Trades': total_trades
            }

        return metrics

    def plot_results(self, symbol: Optional[str] = None):
        """
        Plot comprehensive backtest results including positions

        Parameters:
        -----------
        symbol : Optional[str]
            Symbol to plot results for. If None, plots for all assets.
        """
        if symbol is not None:
            if symbol not in self.results:
                raise ValueError(f"No results for symbol {symbol}")
            symbols = [symbol]
        else:
            symbols = list(self.results.keys())

        for sym in symbols:
            df = self.results[sym]

            # Create a comprehensive figure with 4 subplots
            fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True, gridspec_kw={'height_ratios': [3, 1, 2, 1]})

            # Plot 1: Price with Moving Averages and position highlighting
            ax1 = axs[0]
            ax1.plot(df.index, df['Close'], label=sym, color='blue')

            # Add moving averages if available
            if 'fast_ma' in df.columns and 'slow_ma' in df.columns:
                ax1.plot(df.index, df['fast_ma'], label=f'Fast MA', color='orange', alpha=0.8)
                ax1.plot(df.index, df['slow_ma'], label=f'Slow MA', color='purple', alpha=0.8)

                # If we're using a MovingAverageCrossover strategy, add labels with periods
                if isinstance(self.strategy, MovingAverageCrossover):
                    ax1.plot([], [], label=f'Fast MA ({self.strategy.fast_period} periods)', color='orange')
                    ax1.plot([], [], label=f'Slow MA ({self.strategy.slow_period} periods)', color='purple')

            # Highlight background based on position
            for i in range(1, len(df)):
                pos_value = df['position'].iloc[i]
                if hasattr(pos_value, 'iloc'):  # If it's a Series, get the first value
                    pos_value = pos_value.iloc[0]

                if pos_value > 0:  # Long position
                    ax1.axvspan(df.index[i - 1], df.index[i], alpha=0.1, color='green')
                elif pos_value < 0:  # Short position
                    ax1.axvspan(df.index[i - 1], df.index[i], alpha=0.1, color='red')

            ax1.set_title(f'{sym} Price and Strategy Performance', fontsize=14)
            ax1.set_ylabel('Price', fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(alpha=0.3)

            # Plot 2: Positions
            ax2 = axs[1]
            ax2.fill_between(df.index, df['position'], 0, where=df['position'] > 0, color='green', alpha=0.5,
                             label='Long')
            ax2.fill_between(df.index, df['position'], 0, where=df['position'] < 0, color='red', alpha=0.5,
                             label='Short')
            ax2.plot(df.index, df['position'], color='black', linewidth=0.8)

            # Add markers for position changes
            entries = df.index[(df['position'].shift(1) == 0) & (df['position'] != 0)]
            exits = df.index[(df['position'].shift(1) != 0) & (df['position'] == 0)]

            for entry in entries:
                pos = df.loc[entry, 'position']
                if hasattr(pos, 'iloc'):  # If it's a Series, get the first value
                    pos = pos.iloc[0]

                if pos != 0:  # Only mark non-zero positions
                    color = 'green' if pos > 0 else 'red'
                    marker = '^' if pos > 0 else 'v'
                    ax2.scatter(entry, pos, color=color, s=80, marker=marker, zorder=5)

            for exit in exits:
                ax2.scatter(exit, 0, color='black', s=80, marker='o', zorder=5)

            ax2.set_ylabel('Position', fontsize=12)
            ax2.legend(loc='upper right')
            ax2.grid(alpha=0.3)

            # Plot 3: Strategy Performance
            ax3 = axs[2]
            ax3.plot(df.index, df['cumulative_returns'], label='Buy & Hold', color='gray')
            ax3.plot(df.index, df['strategy_cumulative_returns'], label='Strategy', color='blue')
            ax3.set_ylabel('Cumulative Returns', fontsize=12)
            ax3.legend(loc='upper left')
            ax3.grid(alpha=0.3)

            # Plot 4: Drawdowns
            ax4 = axs[3]
            ax4.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.5)
            ax4.set_ylabel('Drawdown', fontsize=12)
            ax4.set_xlabel('Date', fontsize=12)
            ax4.grid(alpha=0.3)

            # If we have date index, add vertical lines for month boundaries
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 30:
                # The safer way to identify month changes
                dates = pd.Series(df.index)
                months = dates.dt.month
                month_changes = []

                # Find the first day of each month
                for i in range(1, len(months)):
                    if months.iloc[i] != months.iloc[i - 1]:
                        month_changes.append(df.index[i])

                for month_change in month_changes:
                    for ax in axs:
                        ax.axvline(x=month_change, color='gray', linestyle='-', alpha=0.3)

            plt.tight_layout()
            plt.show()

            # Create a second plot with daily summary
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 5:
                # Check if we have intraday data
                has_intraday = False
                try:
                    has_intraday = df.index.hour.any() or df.index.minute.any()
                except AttributeError:
                    # Some datetime indices might not have hour/minute
                    pass

                if has_intraday:
                    try:
                        # Daily position and price
                        daily_positions = df['position'].resample('D').last()
                        daily_close = df['Close'].resample('D').last()

                        # Calculate daily returns
                        daily_returns = df['strategy_returns'].resample('D').sum()

                        # Create the daily summary plots
                        fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

                        # Plot 1: Daily price
                        ax1 = axs[0]
                        ax1.plot(daily_close.index, daily_close, label=f'{sym} Price', color='blue')
                        ax1.set_title(f'{sym} Daily Summary', fontsize=14)
                        ax1.set_ylabel('Price', fontsize=12)
                        ax1.legend()
                        ax1.grid(alpha=0.3)

                        # Plot 2: Daily positions
                        ax2 = axs[1]

                        # Handle the case where positions might be Series objects
                        colors = []
                        for p in daily_positions:
                            if hasattr(p, 'iloc'):  # If it's a Series, get the first value
                                p = p.iloc[0]
                            colors.append('green' if p > 0 else 'red' if p < 0 else 'gray')

                        ax2.bar(daily_positions.index, daily_positions, color=colors, alpha=0.7)
                        ax2.set_ylabel('End-of-Day Position', fontsize=12)
                        ax2.grid(alpha=0.3)

                        # Plot 3: Daily returns
                        ax3 = axs[2]

                        # Handle the case where returns might be Series objects
                        colors = []
                        daily_return_values = []

                        for r in daily_returns:
                            if hasattr(r, 'iloc'):  # If it's a Series, get the first value
                                r = r.iloc[0]
                            colors.append('green' if r > 0 else 'red')
                            daily_return_values.append(r * 100)  # Convert to percentage

                        ax3.bar(daily_returns.index, daily_return_values, color=colors, alpha=0.7)
                        ax3.set_ylabel('Daily Return (%)', fontsize=12)
                        ax3.set_xlabel('Date', fontsize=12)
                        ax3.grid(alpha=0.3)

                        plt.tight_layout()
                        plt.show()
                    except Exception as e:
                        print(f"Warning: Could not generate daily summary plot: {e}")


# Example usage with TY futures using 10d and 20d moving average crossing
if __name__ == "__main__":
    # Create asset - using ^TNX as a proxy for 10-year Treasury futures
    # ZN is the 10-year Treasury futures symbol, but we'll use ^TNX for data availability
    ty_futures = Asset(
        symbol='^TNX',  # 10-year Treasury Yield
        asset_type='futures',
        multiplier=1000,  # Each contract represents $1000 times the face value
        margin_requirement=0.05  # 5% margin requirement
    )

    # Create strategy with 10d and 20d moving averages
    ma_strategy = MovingAverageCrossover(fast_period=10, slow_period=20)

    # Create and run backtest
    backtest = BacktestEngine(
        assets=[ty_futures],
        strategy=ma_strategy,
        start_date='2025-01-01',
        end_date='2025-05-14',
        initial_capital=100000
    )

    results = backtest.run()
    metrics = backtest.calculate_metrics()

    # Print metrics
    for symbol, metric_dict in metrics.items():
        print(
            f"\nPerformance Metrics for {symbol} using {ma_strategy.fast_period}d/{ma_strategy.slow_period}d MA Crossover:")
        for key, value in metric_dict.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    # Plot results
    backtest.plot_results()


