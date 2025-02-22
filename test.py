import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ======== 参数配置 ========
SYMBOL = 'ZN=F'  # 10年期国债期货代码
START_DATE = '2020-01-01'
END_DATE = '2023-01-01'
THRESHOLD_PCT = 2  # 趋势转折阈值（百分比）
WINDOW_SIZE = 5  # 局部极值检测窗口
PRICE_COLUMN = 'Close'  # 使用收盘价分析

# ======== 中文显示设置 ========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ======== 数据获取 ========
def fetch_data():
    """从Yahoo Finance获取历史数据"""
    df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)
    df = df[[PRICE_COLUMN]].dropna()
    print(f"获取到{len(df)}条历史数据（{START_DATE} 至 {END_DATE}）")
    return df


# ======== 自定义趋势检测 ========
def find_pivots(prices, threshold=2, window=5):
    """自定义趋势转折点检测算法"""
    values = prices.values
    dates = prices.index
    length = len(values)

    pivots = np.zeros(length, dtype=int)  # 1=峰值，-1=谷值
    last_pivot = values[0]
    last_type = 0  # 0-初始，1-峰值，-1-谷值

    for i in range(window, length - window):
        # 检测局部极值
        is_peak = np.all(values[i] >= values[i - window:i + window])
        is_valley = np.all(values[i] <= values[i - window:i + window])

        if is_peak or is_valley:
            change_pct = abs((values[i] - last_pivot) / last_pivot * 100)

            if change_pct >= threshold:
                if is_peak and (last_type != 1):
                    pivots[i] = 1
                    last_pivot = values[i]
                    last_type = 1
                elif is_valley and (last_type != -1):
                    pivots[i] = -1
                    last_pivot = values[i]
                    last_type = -1

    idx = np.where(pivots != 0)[0]
    return dates[idx], values[idx], pivots[idx]


# ======== 数据分析 ========
def analyze_swings(dates, levels, types):
    durations = []
    distances = []
    swing_types = []

    for i in range(1, len(dates)):
        delta = dates[i] - dates[i - 1]
        durations.append(int(delta.days))
        distances.append(float(abs(levels[i] - levels[i - 1])))
        swing_types.append('上升' if types[i] == 1 else '下降')

        # 验证所有数组长度一致
        assert len(durations) == len(distances) == len(swing_types), "数组长度不一致"

        # 显式转换为1维数组
    return pd.DataFrame({
        '持续时间': np.array(durations).reshape(-1),
        '价格变动': np.array(distances).reshape(-1),
        '趋势类型': np.array(swing_types).reshape(-1),
        '趋势强度': (np.array(distances) / np.array(durations)).reshape(-1)
    })


# ======== 增强可视化 ========
def plot_analysis(df, dates, levels, types, stats_df):
    """综合可视化分析"""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 4)

    # 主价格走势图
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df[PRICE_COLUMN], label='价格走势', lw=1, alpha=0.7)

    # 标注趋势段
    for i in range(len(dates) - 1):
        start, end = dates[i], dates[i + 1]
        segment = df.loc[start:end]
        color = 'forestgreen' if types[i + 1] == 1 else 'crimson'
        ax1.plot(segment[PRICE_COLUMN], c=color, lw=1.5)

    # 标注转折点
    peak_mask = types == 1
    valley_mask = types == -1
    ax1.scatter(dates[peak_mask], levels[peak_mask], c='lime', s=80,
                edgecolor='black', label='峰值点', zorder=5)
    ax1.scatter(dates[valley_mask], levels[valley_mask], c='red', s=80,
                edgecolor='black', label='谷值点', zorder=5)
    ax1.set_title('价格趋势阶段分析', fontsize=14)
    ax1.legend()

    # 持续时间分析
    ax2 = fig.add_subplot(gs[1, 0])
    stats_df['持续时间'].plot(kind='box', vert=False, ax=ax2, color='teal')
    ax2.set_xlabel('持续时间（天）')
    ax2.set_title('持续时间分布')

    # 价格变动分析
    ax3 = fig.add_subplot(gs[1, 1])
    stats_df['价格变动'].plot(kind='hist', bins=15, ax=ax3,
                              color='salmon', alpha=0.7)
    ax3.set_xlabel('价格变动')
    ax3.set_title('价格波动分布')

    # 趋势强度矩阵
    ax4 = fig.add_subplot(gs[1, 2])
    scatter = ax4.scatter(stats_df['持续时间'], stats_df['价格变动'],
                          c=stats_df['趋势强度'], cmap='viridis', s=50)
    plt.colorbar(scatter, label='趋势强度（价格变动/天）')
    ax4.set_xlabel('持续时间')
    ax4.set_ylabel('价格变动')
    ax4.set_title('趋势强度矩阵')
    ax4.grid(True)

    # 趋势强度分布
    ax5 = fig.add_subplot(gs[1, 3])
    stats_df['趋势强度'].plot(kind='kde', ax=ax5, color='purple')
    ax5.set_xlabel('趋势强度')
    ax5.set_title('趋势强度分布')

    # 时间序列分析
    ax6 = fig.add_subplot(gs[2, :])
    stats_df['持续时间'].plot(kind='bar', ax=ax6, color='skyblue',
                              alpha=0.7, label='持续时间')
    ax6b = ax6.twinx()
    stats_df['趋势强度'].plot(kind='line', ax=ax6b, color='darkorange',
                              lw=2, marker='o', label='趋势强度')
    ax6.set_xlabel('趋势段序号')
    ax6.set_ylabel('持续时间（天）')
    ax6b.set_ylabel('趋势强度')
    ax6.set_title('趋势演变分析')
    ax6.legend(loc='upper left')
    ax6b.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


# ======== 主程序 ========
if __name__ == "__main__":
    # 获取数据
    df = fetch_data()

    # 检测趋势转折点
    dates, levels, types = find_pivots(df[PRICE_COLUMN], THRESHOLD_PCT, WINDOW_SIZE)
    print(f"\n检测到{len(dates)}个趋势转折点")

    # 计算统计数据
    stats_df = analyze_swings(dates, levels, types)

    # 显示统计摘要
    print("\n全局统计摘要：")
    print(stats_df.describe())

    # 分类统计
    print("\n分类统计摘要：")
    print(stats_df.groupby('趋势类型').describe())

    # 可视化分析
    plot_analysis(df, dates, levels, types, stats_df)