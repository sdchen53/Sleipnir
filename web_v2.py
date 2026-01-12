import streamlit as st
import hashlib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import glob
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ta
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


from myLogger import myLogger
from utils.load import getStockMarketData
from utils.workspace import auth, generateNewWorkspace, generateNewThread, chatWithThread
from utils.operation import extract_prediction_line, extract_confidence_operation


# Configure the page
st.set_page_config(page_title="Secure App", page_icon="🔒")

# Password configuration (in production, store this securely)
CORRECT_PASSWORD_HASH = hashlib.sha256("mypassword123@123".encode()).hexdigest()

def check_password(password):
    """Check if the provided password is correct"""
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == CORRECT_PASSWORD_HASH

def login_page():
    """Display login page"""
    st.title("🔒 Login Required")
    st.write("Please enter your password to access the application.")
    
    password = st.text_input("Password", type="password", key="password_input")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        login_button = st.button("Login", use_container_width=True)
    
    if login_button:
        if check_password(password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
            st.info("💡 Hint: Default password is 'mypassword123'")


# ==== 股票预测与结果分析函数 BEGIN ====
def sync_data_by_date_range(df_market, df_operation):
    """
    Get date range from df_operation and sync with df_market data
    If date exists in df_market but not in df_operation, add it to df_operation with 0 values
    Returns filtered dataframes with common date range
    """
    df_market = df_market.sort_index()
    operation_start_date = df_operation.index.min()
    operation_end_date = df_operation.index.max()
    # print(f"Operation date range: {operation_start_date} to {operation_end_date}")
    df_market_filtered = df_market.loc[operation_start_date:operation_end_date]
    # print(f"Market data entries in date range: {len(df_market_filtered)}")
    # print(f"Operation data entries: {len(df_operation)}")
    df_operation_synced = df_operation.reindex(df_market_filtered.index, fill_value=0)
    df_market_synced = df_market_filtered
    # print(f"After synchronization:")
    # print(f"Number of dates: {len(df_market_synced)}")
    # print(f"Date range: {df_market_synced.index.min()} to {df_market_synced.index.max()}")
    return df_market_synced, df_operation_synced

def calculate_returns(df_market_synced, df_operation_synced, initial_capital=1000):
    """
    Calculate returns based on trading operations with high initial capital
    Decision: 1 = buy, -1 = sell, 0 = hold
    Hands: number of shares to trade
    """
    df_combined = df_market_synced.copy()
    df_combined['Decision'] = pd.to_numeric(df_operation_synced['Decision'], errors='coerce').fillna(0).astype(int)
    df_combined['Hands'] = pd.to_numeric(df_operation_synced['Hands'], errors='coerce').fillna(0).astype(int)
    initial_capital = initial_capital * df_combined['close'][0]
    portfolio_values = []
    cash = float(initial_capital)
    shares_held = int(0)
    trade_log = []
    for i, (date, row) in enumerate(df_combined.iterrows()):
        price = float(row['close'])
        decision = int(row['Decision'])
        hands = int(row['Hands'])
        if decision == 1 and hands > 0:
            trade_cost = float(hands) * price
            if cash >= trade_cost:
                cash = cash - trade_cost
                shares_held = shares_held + hands
                trade_log.append({
                    'date': date,
                    'action': 'BUY',
                    'shares': hands,
                    'price': price,
                    'value': trade_cost,
                    'cash_after': cash,
                    'shares_after': shares_held
                })
            else:
                affordable_shares = int(cash / price)
                if affordable_shares > 0:
                    actual_cost = float(affordable_shares) * price
                    cash = cash - actual_cost
                    shares_held = shares_held + affordable_shares
                    trade_log.append({
                        'date': date,
                        'action': 'PARTIAL_BUY',
                        'shares': affordable_shares,
                        'price': price,
                        'value': actual_cost,
                        'cash_after': cash,
                        'shares_after': shares_held
                    })
        elif decision == -1 and hands > 0:
            shares_to_sell = min(hands, shares_held)
            if shares_to_sell > 0:
                trade_value = float(shares_to_sell) * price
                cash = cash + trade_value
                shares_held = shares_held - shares_to_sell
                trade_log.append({
                    'date': date,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': price,
                    'value': trade_value,
                    'cash_after': cash,
                    'shares_after': shares_held
                })
        current_portfolio_value = cash + (float(shares_held) * price)
        portfolio_values.append(current_portfolio_value)
    df_combined['Portfolio_Value'] = portfolio_values
    df_combined['Cash'] = cash
    df_combined['Shares_Held'] = shares_held
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    df_combined['Daily_Return'] = df_combined['Portfolio_Value'].pct_change()
    df_combined['Cumulative_Return'] = (df_combined['Portfolio_Value'] / initial_capital) - 1
    initial_price = float(df_combined['close'].iloc[0])
    final_price = float(df_combined['close'].iloc[-1])
    buy_hold_return = (final_price - initial_price) / initial_price
    performance_metrics = {
        'Initial Capital': initial_capital,
        'Final Portfolio Value': final_value,
        'Total Return': total_return,
        'Total Return (%)': total_return * 100,
        'Buy & Hold Return (%)': buy_hold_return * 100,
        'Excess Return (%)': (total_return - buy_hold_return) * 100,
        'Number of Trades': len(trade_log),
        'Final Cash': cash,
        'Final Shares': shares_held,
        'Final Stock Price': final_price,
        'Annualized Return (%)': (((final_value / initial_capital) ** (252 / len(df_combined))) - 1) * 100,
    }
    if len(df_combined['Daily_Return'].dropna()) > 1:
        daily_returns = df_combined['Daily_Return'].dropna()
        performance_metrics.update({
            'Volatility (%)': daily_returns.std() * np.sqrt(252) * 100,
            'Sharpe Ratio': (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0,
            'Max Drawdown (%)': ((df_combined['Portfolio_Value'] / df_combined['Portfolio_Value'].cummax()) - 1).min() * 100
        })
    return df_combined, pd.DataFrame(trade_log), performance_metrics

def plot_performance_analysis(df_results, metrics):
    """
    Create comprehensive performance visualization
    """
    plt.style.use('default')
    sns.set_palette("husl")
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Trading Strategy Performance Analysis', fontsize=16, fontweight='bold')
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(df_results.index, df_results['Portfolio_Value'], 'b-', label='Portfolio Value', linewidth=2.5, alpha=0.8)
    ax1.set_ylabel('Portfolio Value ($)', color='b', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B' if x >= 1e9 else f'${x/1e6:.1f}M'))
    line2 = ax1_twin.plot(df_results.index, df_results['close'], 'r--', label='Stock Price', linewidth=2, alpha=0.7)
    ax1_twin.set_ylabel('Stock Price ($)', color='r', fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Portfolio Value vs Stock Price', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    strategy_returns = df_results['Cumulative_Return'] * 100
    buy_hold_returns = (df_results['close'] / df_results['close'].iloc[0] - 1) * 100
    ax2.plot(df_results.index, strategy_returns, 'g-', label='Strategy Return', linewidth=2.5, alpha=0.8)
    ax2.plot(df_results.index, buy_hold_returns, 'orange', label='Buy & Hold Return', linestyle='--', linewidth=2, alpha=0.8)
    ax2.set_ylabel('Cumulative Return (%)', fontweight='bold')
    ax2.set_title('Cumulative Returns Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    textstr = f'Strategy: {metrics["Total Return (%)"]:.2f}%\nBuy&Hold: {metrics["Buy & Hold Return (%)"]:.2f}%\nExcess: {metrics["Excess Return (%)"]:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.62, 0.98, textstr, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    ax3.plot(df_results.index, df_results['close'], 'k-', alpha=0.6, label='Stock Price', linewidth=1.5)
    buy_dates = df_results[df_results['Decision'] == 1].index
    sell_dates = df_results[df_results['Decision'] == -1].index
    if len(buy_dates) > 0:
        ax3.scatter(buy_dates, df_results.loc[buy_dates, 'close'], color='green', marker='^', s=100, label='Buy', alpha=0.8, edgecolors='darkgreen')
    if len(sell_dates) > 0:
        ax3.scatter(sell_dates, df_results.loc[sell_dates, 'close'], color='red', marker='v', s=100, label='Sell', alpha=0.8, edgecolors='darkred')
    ax3.set_ylabel('Stock Price ($)', fontweight='bold')
    ax3.set_title('Trading Activity', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    stock_values = df_results['Shares_Held'] * df_results['close']
    cash_values = df_results['Portfolio_Value'] - stock_values
    ax4.fill_between(df_results.index, 0, cash_values, label='Cash', alpha=0.6, color='lightblue')
    ax4.fill_between(df_results.index, cash_values, df_results['Portfolio_Value'], label='Stock Holdings', alpha=0.6, color='lightcoral')
    ax4.set_ylabel('Value ($)', fontweight='bold')
    ax4.set_title('Portfolio Composition', fontweight='bold')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B' if x >= 1e9 else f'${x/1e6:.1f}M'))
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    # 打印详细指标
    formatted_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'Capital' in key or 'Value' in key or 'Cash' in key or 'Price' in key:
                if value >= 1e9:
                    formatted_metrics[key] = f"${value/1e9:.2f}B"
                elif value >= 1e6:
                    formatted_metrics[key] = f"${value/1e6:.2f}M"
                else:
                    formatted_metrics[key] = f"${value:,.2f}"
            elif '%' in key or 'Return' in key or 'Ratio' in key:
                formatted_metrics[key] = f"{value:.4f}" if 'Ratio' in key else f"{value:.2f}%"
            else:
                formatted_metrics[key] = f"{value:.2f}"
        else:
            formatted_metrics[key] = str(value)
    metrics_df = pd.DataFrame(list(formatted_metrics.items()), columns=['Metric', 'Value'])
    print(metrics_df.to_string(index=False))
    return fig
# ==== 股票预测与结果分析函数 END ====

quantitive_factor = """
You are an AI investment advisor specializing in small-cap value strategies, providing insights and predictions based on historical data.

## Investment Focus
- Small-cap stocks (market cap < 1B USD)
- Value factors: low P/E, P/B, P/S ratios
- Quality metrics: ROE, ROA, debt levels

## Decision Framework
**BUY Signals:**
- Strong value metrics vs peers/history
- Solid fundamentals despite small size
- Positive alpha potential

**SELL Signals:**
- Valuation reaches fair value
- Deteriorating fundamentals
- Better opportunities available

## Response Format
**Stock**: [Ticker/Name]
**Price**: [Current] | **Market Cap**: [Value]

**Scores** (1-10):
- Size Factor: [Score]
- Value Factor: [Score] 
- Quality Factor: [Score]

**Decision**: [BUY/HOLD/SELL]
**Confidence**: [1-10]
**Target Allocation**: [%]
**Expected Return**: [% over timeframe]

**Rationale**: [1-2 key points]
**Key Risk**: [Main concern]
**Entry/Exit**: [Price levels]

Note: Size factor effectiveness in A-shares has weakened post-2017.
"""

bollinger_bands = """
You are a Bollinger Bands mean reversion trader. Your strategy: buy when price breaks below lower band, sell when price breaks above upper band, exit at middle band.

## Setup
- Period: 20 days
- Standard deviation: 2.0
- Upper Band = MA + (2 x StdDev)
- Lower Band = MA - (2 x StdDev)

## Signals
**BUY**: Price closes below lower band
**SELL**: Price closes above upper band  
**EXIT**: Price returns to middle band or 3% stop-loss

## Decision Format
**Symbol**: [Ticker]
**Signal**: [BUY/SELL/EXIT/HOLD]
**Current Price**: [Price]
**Band Position**: [Above/Below/Within bands]
**Entry/Exit Price**: [Specific level]
**Stop Loss**: [3% from entry]
**Rationale**: [1-2 sentences]

## Risk Rules
- Max 5% portfolio per trade
- No trading in strong trends
- Exit if no reversion within 10 days

Keep responses brief and actionable.
"""

camp_model = """
You are a CAPM model analyst. Your role is to determine when CAPM is appropriate for asset pricing and when multi-factor models are needed.

## CAPM Framework
**Formula**: E(Ri) = Rf + βi(E(Rm) - Rf)
- Ri = Expected return of asset i
- Rf = Risk-free rate
- βi = Beta (systematic risk)
- Rm = Market return

## Decision Criteria

**Use CAPM When:**
- Analyzing broad market portfolios
- Beta explains >70% of return variance
- No significant market anomalies present
- Short-term analysis (< 1 year)

**Use Multi-Factor Models When:**
- Significant alpha detected (α ≠ 0)
- Market anomalies present (size, value effects)
- Individual stock analysis
- Long-term analysis (> 1 year)

## Response Format
**Asset**: [Name/Ticker]
**Beta**: [Value]
**R-squared**: [% variance explained by market]

**Model Recommendation**: [CAPM/Multi-Factor]
**Reasoning**: [1-2 sentences why]

**If Multi-Factor Needed:**
- Additional factors: [Size/Value/Momentum/etc.]
- Expected alpha: [%]

**CAPM Limitations**: [Key assumptions violated]
"""


default_character_presets = {
    "base": "You are an AI investment advisor specializing in small-cap value strategies, providing insights and predictions based on historical data.",
    "quantitive_factor": quantitive_factor,
    "bollinger_bands": bollinger_bands,
    "camp_model": camp_model,
    # "final": "You are a final decision maker, aggregating insights from multiple agents to provide a comprehensive investment strategy."
    }
leader_preset = "You are a leader decision maker, aggregating insights from multiple agents to provide a comprehensive investment strategy."
anything_api = "1DV9A3A-SFFM1XR-QF4TYMR-HZ5X8RY"
entrypoint = "http://10.201.35.124:3001/api/v1/"
message_preset = "@agent Get {} stock info, from {} to {}, predict the later day's price, and give the buy-in or hold or sell-out decision on {}, with confidence."
message_outformat = "Current holding is {} shares, max holding is {} shares. Trade limit per operation is{}, expected return percentage is {}. Answer MUST contain Example style: '{}, buy-in, 0.5, hold, 0.1, sell-out, 0.4, hands-in, 200, hands-off, 100'"

@st.cache_data
def get_authed():
        # auth
    try:
        auth_response = auth()
        print(f"Authentication successful: {auth_response}")
    except Exception as e:
        print(f"Error during authentication: {e}")
        sys.exit(1)

    try:
        workspace_name = "My New Workspace"
        workspace_slug = generateNewWorkspace(workspace_name)
        print(f"Workspace created successfully: {workspace_slug}")
        return True, workspace_slug
    except Exception as e:
        print(f"Error during workspace creation: {e}")
        sys.exit(1)

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_stock_data():
    """Load all stock data from CSV files in the data directory"""
    data_files = glob.glob("./data/stock_market_data-*.csv")
    
    if not data_files:
        st.error("No stock data files found in ./data directory!")
        return {}
    
    stocks_data = {}
    
    for file in data_files:
        try:
            # Extract stock symbol from filename
            symbol = os.path.basename(file).replace("stock_market_data-", "").replace(".csv", "")
            
            # Load the CSV file
            df = pd.read_csv(file)
            
            # Ensure proper column names and data types
            df.columns = df.columns.str.lower()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df.reset_index(drop=True, inplace=True)
            
            # Calculate additional technical indicators
            df = calculate_technical_indicators(df)
            
            stocks_data[symbol] = df
            
        except Exception as e:
            st.warning(f"Error loading {file}: {str(e)}")
    
    return stocks_data

def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    try:
        # Basic price indicators
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'])
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Volume indicators (only if volume column exists and has valid data)
        if 'volume' in df.columns and df['volume'].notna().any():
            try:
                # Volume Weighted Average Price
                df['vwap'] = ta.volume.VolumePriceTrendIndicator(
                    close=df['close'], volume=df['volume']
                ).volume_price_trend()
                
                # On Balance Volume
                df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                    close=df['close'], volume=df['volume']
                ).on_balance_volume()
                
                # Accumulation/Distribution Line
                df['ad'] = ta.volume.AccDistIndexIndicator(
                    high=df['high'], low=df['low'], 
                    close=df['close'], volume=df['volume']
                ).acc_dist_index()
                
            except Exception as e:
                print(f"Error calculating volume indicators: {e}")
        
        # Price change and returns - FIXED: Calculate these properly
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['close'].diff()
        df['daily_return'] = df['close'].pct_change() * 100  # Percentage returns
        
        # Fill NaN values for the first row
        df['price_change'] = df['price_change'].fillna(0)
        df['price_change_abs'] = df['price_change_abs'].fillna(0)
        df['daily_return'] = df['daily_return'].fillna(0)
        
    except Exception as e:
        st.warning(f"Error calculating technical indicators: {str(e)}")
        # Ensure basic columns exist even if calculation fails
        if 'price_change' not in df.columns:
            df['price_change'] = df['close'].pct_change().fillna(0)
        if 'daily_return' not in df.columns:
            df['daily_return'] = df['close'].pct_change().fillna(0) * 100
    
    return df

def create_candlestick_chart(df, symbol, show_volume=True, show_indicators=True):
    """Create an interactive candlestick chart with technical indicators"""
    
    # Determine number of subplots
    rows = 3 if show_volume else 2
    subplot_titles = ['Price Chart', 'RSI', 'Volume'] if show_volume else ['Price Chart', 'RSI']
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_width=[0.6, 0.2, 0.2] if show_volume else [0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=f'{symbol} Price',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000'
        ),
        row=1, col=1
    )
    
    if show_indicators:
        # Moving averages (only show if they exist and have data)
        if 'sma_20' in df.columns and df['sma_20'].notna().any():
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['sma_20'], 
                          name='SMA 20', line=dict(color='orange', width=1)),
                row=1, col=1
            )
        
        if 'sma_50' in df.columns and df['sma_50'].notna().any():
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['sma_50'], 
                          name='SMA 50', line=dict(color='blue', width=1)),
                row=1, col=1
            )
        
        # Bollinger Bands (only show if they exist and have data)
        if 'bb_upper' in df.columns and df['bb_upper'].notna().any():
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['bb_upper'], 
                          name='BB Upper', line=dict(color='gray', width=1, dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['bb_lower'], 
                          name='BB Lower', line=dict(color='gray', width=1, dash='dash')),
                row=1, col=1
            )
    
    # RSI (only show if it exists and has data)
    if 'rsi' in df.columns and df['rsi'].notna().any():
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['rsi'], 
                      name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume (only show if requested and volume data exists)
    if show_volume and 'volume' in df.columns and df['volume'].notna().any():
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['volume'], 
                  name='Volume', marker_color=colors, opacity=0.7),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Analysis',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def create_correlation_matrix(stocks_data):
    """Create correlation matrix of selected stocks"""
    if len(stocks_data) < 2:
        return None
    
    # Get closing prices for all stocks
    price_data = {}
    for symbol, df in stocks_data.items():
        price_data[symbol] = df.set_index('timestamp')['close']
    
    # Create DataFrame with aligned dates
    combined_df = pd.DataFrame(price_data)
    
    # Calculate correlation matrix
    correlation_matrix = combined_df.corr()
    
    # Create heatmap
    fig = px.imshow(
        correlation_matrix.values,
        labels=dict(x="Stock", y="Stock", color="Correlation"),
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        color_continuous_scale='RdBu',
        aspect="auto",
        title="Stock Price Correlation Matrix"
    )
    
    return fig

def calculate_portfolio_metrics(df):
    """Calculate key portfolio metrics"""
    try:
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        # Calculate volatility (standard deviation of returns)
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0  # Annualized volatility
        
        # Calculate max drawdown
        rolling_max = df['close'].expanding().max()
        drawdown = (df['close'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        if len(returns) > 0 and returns.std() != 0:
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        return {
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {}

def predict_price(df, days_ahead=5):
    """Simple price prediction using Random Forest"""
    try:
        # Prepare features
        features = ['open', 'high', 'low', 'volume', 'sma_20', 'sma_50', 'rsi']
        available_features = [f for f in features if f in df.columns and df[f].notna().any()]
        
        if len(available_features) < 3:
            return None
        
        # Remove rows with NaN values
        clean_df = df[['close'] + available_features].dropna()
        
        if len(clean_df) < 50:  # Need enough data for training
            return None
        
        X = clean_df[available_features].values
        y = clean_df['close'].values
        
        # Use last 80% for training
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make prediction for next period
        last_features = X_test_scaled[-1].reshape(1, -1)
        prediction = model.predict(last_features)[0]
        
        return prediction
        
    except Exception as e:
        st.warning(f"Error in price prediction: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">📈 Advanced Stock Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading stock data..."):
        stocks_data = load_stock_data()
    
    if not stocks_data:
        st.error("No stock data available. Please check your data directory.")
        return
    
    # Sidebar controls
    st.sidebar.header("📊 Analysis Controls")
    
    # Stock selection
    available_stocks = list(stocks_data.keys())
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks for Analysis",
        available_stocks,
        default=available_stocks[:min(3, len(available_stocks))]
    )
    
    if not selected_stocks:
        st.warning("Please select at least one stock for analysis.")
        return
    
    # Date range selection
    all_dates = []
    for stock in selected_stocks:
        all_dates.extend(stocks_data[stock]['timestamp'].tolist())
    
    min_date = min(all_dates).date()
    max_date = max(all_dates).date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(max_date - timedelta(days=365), max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Chart options
    st.sidebar.subheader("Chart Options")
    show_volume = st.sidebar.checkbox("Show Volume", value=True)
    show_indicators = st.sidebar.checkbox("Show Technical Indicators", value=True)
    
    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Individual Stock Analysis", "Portfolio Comparison", "Correlation Analysis", "Prediction", "Agent Trading"]
    )
    
    # Filter data by date range
    filtered_data = {}
    for stock in selected_stocks:
        df = stocks_data[stock]
        mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
        filtered_data[stock] = df.loc[mask].copy()
    
    # Main content based on analysis type
    if analysis_type == "Individual Stock Analysis":
        primary_stock = st.selectbox("Select Primary Stock", selected_stocks)
        df = filtered_data[primary_stock]
        
        if df.empty:
            st.warning("No data available for the selected date range.")
            return
        
        # Key metrics
        metrics = calculate_portfolio_metrics(df)
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${metrics['current_price']:.2f}",
                    f"{metrics['price_change_pct']:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "Volatility",
                    f"{metrics['volatility']:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Max Drawdown",
                    f"{metrics['max_drawdown']:.2f}%"
                )
            
            with col4:
                st.metric(
                    "Sharpe Ratio",
                    f"{metrics['sharpe_ratio']:.2f}"
                )
        
        # Main chart
        fig = create_candlestick_chart(df, primary_stock, show_volume, show_indicators)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Distribution")
            fig_hist = px.histogram(df, x='close', nbins=50, title="Price Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("Daily Returns")
            # FIXED: Check if price_change column exists and has data
            if 'daily_return' in df.columns and df['daily_return'].notna().any():
                fig_returns = px.histogram(df, x='daily_return', nbins=50, title="Daily Returns Distribution (%)")
                st.plotly_chart(fig_returns, use_container_width=True)
            else:
                # Fallback: calculate returns on the fly
                daily_returns = df['close'].pct_change().fillna(0) * 100
                fig_returns = px.histogram(x=daily_returns, nbins=50, title="Daily Returns Distribution (%)")
                st.plotly_chart(fig_returns, use_container_width=True)
    
    elif analysis_type == "Portfolio Comparison":
        st.subheader("Portfolio Performance Comparison")
        
        # Normalize prices to show percentage changes
        comparison_data = {}
        for stock in selected_stocks:
            df = filtered_data[stock]
            if not df.empty:
                df = df.copy()
                df['normalized_price'] = (df['close'] / df['close'].iloc[0]) * 100
                comparison_data[stock] = df
        
        # Create comparison chart
        fig = go.Figure()
        
        for stock, df in comparison_data.items():
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['normalized_price'],
                    name=stock,
                    mode='lines'
                )
            )
        
        fig.update_layout(
            title="Normalized Price Comparison (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            height=600,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics table
        st.subheader("Performance Metrics Comparison")
        
        metrics_data = []
        for stock in selected_stocks:
            df = filtered_data[stock]
            if not df.empty:
                metrics = calculate_portfolio_metrics(df)
                if metrics:
                    metrics_data.append({
                        'Stock': stock,
                        'Current Price': f"${metrics['current_price']:.2f}",
                        'Change %': f"{metrics['price_change_pct']:+.2f}%",
                        'Volatility %': f"{metrics['volatility']:.2f}%",
                        'Max Drawdown %': f"{metrics['max_drawdown']:.2f}%",
                        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}"
                    })
        
        if metrics_data:
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    elif analysis_type == "Correlation Analysis":
        st.subheader("Stock Correlation Analysis")
        
        fig_corr = create_correlation_matrix(filtered_data)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Need at least 2 stocks for correlation analysis.")
    
    elif analysis_type == "Prediction":
        st.subheader("Price Prediction (ML-based)")
        
        primary_stock = st.selectbox("Select Stock for Prediction", selected_stocks)
        df = filtered_data[primary_stock]
        
        if not df.empty:
            prediction = predict_price(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                current_price = df['close'].iloc[-1]
                st.metric("Current Price", f"${current_price:.2f}")
                
                if prediction:
                    change = prediction - current_price
                    change_pct = (change / current_price) * 100
                    st.metric(
                        "Predicted Next Price",
                        f"${prediction:.2f}",
                        f"{change_pct:+.2f}%"
                    )
                else:
                    st.warning("Unable to generate prediction. Insufficient data.")
            
            with col2:
                # Show recent price trend
                recent_df = df.tail(30)
                fig_recent = px.line(
                    recent_df, x='timestamp', y='close',
                    title="Recent Price Trend (Last 30 periods)"
                )
                st.plotly_chart(fig_recent, use_container_width=True)

    elif analysis_type == "Agent Trading":
        st.subheader("Agent Decision Making for Stock Trading")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.selectbox("Select Stock for Trading", selected_stocks)
            start_date = st.date_input('Start date', value=pd.to_datetime('2023-01-01'))
            end_date = st.date_input('End date', value=pd.to_datetime('2023-01-31'))
            lookback_days = st.number_input('Lookback days', min_value=1, value=30, step=1)
        
        with col2:
            current_holding = st.number_input('Current holding shares', min_value=0, value=500, step=100)
            max_holding = st.number_input('Max holding shares', min_value=100, value=1000, step=100)
            trade_limit = st.number_input('Trade limit per operation', min_value=100, value=300, step=100)
            return_expectation = st.number_input('Expected return percentage', min_value=0.05, value=0.1, step=0.05)
        
        # Agent character selection
        character_options = list(default_character_presets.keys())
        selected_characters = st.multiselect(
            'Select agent characters to use',
            options=character_options,
            default=character_options
        )
        character_presets = {char: default_character_presets[char] for char in selected_characters}
        
        # Get stock data
        df_stock = getStockMarketData(ticker)
        daterange = pd.date_range(start=start_date, end=end_date, freq='B')
        df_filtered = df_stock[df_stock.index.isin(daterange)]
        daterange = sorted(df_filtered.index)
        
        # Authentication
        authed, workspace_slug = get_authed()
        if not authed:
            st.warning("⚠️ Please authenticate first by clicking 'Create Agents' button.")
        else:
            st.info(f"✅ Using workspace: {workspace_slug}")
        
        logger = myLogger(name=str(os.getpid()), log_filename=workspace_slug, propagate=False)
        
        # Agent creation section
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button('🤖 Create Agents', use_container_width=True):
                if not authed:
                    st.error("Please authenticate first.")
                    st.stop()
                
                with st.spinner('Creating agent threads...'):
                    st.session_state.character_slugs = {}
                    
                    # Create progress bar for agent creation
                    agent_progress = st.progress(0)
                    total_agents = len(character_presets) + 1  # +1 for leader
                    
                    for idx, (character, preset) in enumerate(character_presets.items()):
                        try:
                            logger(f"Creating thread for character: {character}")
                            thread_slug = generateNewThread(workspace_slug)
                            logger(f"Thread created successfully: {thread_slug}")
                            st.session_state.character_slugs[character] = thread_slug
                            
                            # Initialize agent
                            message = f"@agent {preset}"
                            chat_response = chatWithThread(workspace_slug, thread_slug, message)
                            logger(f"Chat response: {chat_response}")
                            
                            agent_progress.progress((idx + 1) / total_agents)
                        except Exception as e:
                            logger.error_(f"Error during thread creation: {e}")
                            st.error(f"Error creating {character}: {str(e)}")
                    
                    # Create leader thread
                    try:
                        st.session_state.leader_thread_slug = generateNewThread(workspace_slug)
                        logger(f"Leader thread created successfully: {st.session_state.leader_thread_slug}")
                        
                        message = f"@agent {leader_preset}"
                        chat_response = chatWithThread(workspace_slug, st.session_state.leader_thread_slug, message)
                        logger(f"Chat response: {chat_response}")
                        
                        agent_progress.progress(1.0)
                        st.success("✅ Agents created and initialized successfully.")
                    except Exception as e:
                        logger.error_(f"Error during final thread creation: {e}")
                        st.error(f"Error creating leader: {str(e)}")
        
        with col_btn2:
            if st.button('📈 Start Trading Decisions', use_container_width=True):
                if 'character_slugs' not in st.session_state or 'leader_thread_slug' not in st.session_state:
                    st.error("Please create agents first!")
                    st.stop()
                
                # Initialize results
                pd_results = pd.DataFrame(columns=['timestamp', 'Decision', 'Hands'])
                results_placeholder = st.empty()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                temp_holding = current_holding
                
                for idx, date in enumerate(daterange):
                    date_str = date.strftime('%Y-%m-%d')
                    status_text.text(f"Processing {date_str}... ({idx+1}/{len(daterange)})")
                    
                    lookback_start_date = (date - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                    
                    message = message_preset.format(
                        ticker, lookback_start_date, date_str, date_str
                    ) + message_outformat.format(
                        temp_holding, max_holding, trade_limit, return_expectation,
                        f"{ticker}, {date_str}"
                    )
                    
                    try:
                        # Collect responses from all characters
                        response_list = []
                        for character, thread_slug in st.session_state.character_slugs.items():
                            try:
                                chat_response = chatWithThread(workspace_slug, thread_slug, message)
                                logger(f"Chat response from {character}: {chat_response}")
                                response_list.append(chat_response)
                            except Exception as e:
                                logger.error_(f"Error during chat with {character}: {e}")
                                response_list.append(f"Error from {character}: {str(e)}")
                        
                        # Aggregate and get final decision
                        aggregated_response = "\n".join(response_list)
                        chat_response = chatWithThread(
                            workspace_slug, 
                            st.session_state.leader_thread_slug,
                            aggregated_response + message_outformat.format(
                                temp_holding, max_holding, trade_limit, 
                                return_expectation, f"{ticker}, {date_str}"
                            )
                        )
                        
                        one_line = extract_prediction_line(chat_response)
                        decision, hands = extract_confidence_operation(one_line)
                        logger(f"Decision on {date_str}: {decision}, Hands: {hands}")
                        
                        temp_holding = max(temp_holding + hands * decision, 0)
                        
                        # Add to results
                        new_result = pd.DataFrame({
                            'timestamp': [date_str],
                            'Decision': [decision],
                            'Hands': [hands]
                        })
                        pd_results = pd.concat([pd_results, new_result], ignore_index=True)
                        
                    except Exception as e:
                        logger.error_(f"Error processing date {date_str}: {e}")
                        error_result = pd.DataFrame({
                            'timestamp': [date_str],
                            'Decision': [0],
                            'Hands': [0]
                        })
                        pd_results = pd.concat([pd_results, error_result], ignore_index=True)
                    
                    # Update display
                    progress_bar.progress((idx + 1) / len(daterange))
                    results_placeholder.dataframe(pd_results, use_container_width=True)
                
                status_text.text("✅ All decisions completed!")
                
                # Summary metrics
                st.subheader("Trading Summary")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    buy_decisions = (pd_results['Decision'] == 1).sum()
                    st.metric("Buy Decisions", buy_decisions)
                
                with metric_col2:
                    sell_decisions = (pd_results['Decision'] == -1).sum()
                    st.metric("Sell Decisions", sell_decisions)
                
                with metric_col3:
                    hold_decisions = (pd_results['Decision'] == 0).sum()
                    st.metric("Hold Decisions", hold_decisions)
                
                # Download button for results
                csv = pd_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"{ticker}_trading_decisions_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This analysis is for educational purposes only and should not be considered as financial advice."
    )

if __name__ == "__main__":
        
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # Show appropriate page based on authentication status
    if st.session_state.authenticated:
        main()
    else:
        login_page()