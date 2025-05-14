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
        Plot backtest results

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

            plt.figure(figsize=(12, 10))

            # Plot 1: Price and Moving Averages
            plt.subplot(3, 1, 1)
            plt.plot(df.index, df['Close'], label=sym)
            if 'fast_ma' in df.columns and 'slow_ma' in df.columns:
                plt.plot(df.index, df['fast_ma'], label=f'{self.strategy.fast_period}-day MA')
                plt.plot(df.index, df['slow_ma'], label=f'{self.strategy.slow_period}-day MA')
            plt.title(f'{sym} Price Chart')
            plt.legend()

            # Plot 2: Strategy Performance
            plt.subplot(3, 1, 2)
            plt.plot(df.index, df['cumulative_returns'], label='Buy & Hold')
            plt.plot(df.index, df['strategy_cumulative_returns'], label='Strategy')
            plt.title('Strategy Performance')
            plt.legend()

            # Plot 3: Drawdowns
            plt.subplot(3, 1, 3)
            plt.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3)
            plt.title('Drawdowns')
            plt.tight_layout()

            plt.show()


# Example usage with TY futures using 10d and 20d moving average crossing
if __name__ == "__main__":
    # Create asset - using ^TNX as a proxy for 10-year Treasury futures
    # ZN is the 10-year Treasury futures symbol, but we'll use ^TNX for data availability
    stock = Asset(
        symbol='AAPL',
        asset_type='Stock',
        multiplier=1000,  # Each contract represents $1000 times the face value
        margin_requirement=0.05  # 5% margin requirement
    )

    # Create strategy with 10d and 20d moving averages
    ma_strategy = MovingAverageCrossover(fast_period=10, slow_period=20)

    # Create and run backtest
    backtest = BacktestEngine(
        assets=[stock],
        strategy=ma_strategy,
        start_date='2020-01-01',
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


