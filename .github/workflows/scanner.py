import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import requests
import json
import os
from duckduckgo_search import DDGS

# ==========================================
# CONFIGURATION
# ==========================================

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

NIFTY_50_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
    'TITAN.NS', 'BAJFINANCE.NS', 'WIPRO.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
    'HCLTECH.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATAMOTORS.NS',
    'TATASTEEL.NS', 'BAJAJFINSV.NS', 'M&M.NS', 'TECHM.NS', 'ADANIPORTS.NS'
]

# ==========================================
# DATA COLLECTION
# ==========================================

def fetch_stock_data(ticker, days=60):
    """Fetch historical stock data"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{days}d")
        return df if len(df) > 30 else None
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def fetch_news(ticker):
    """Fetch latest news for stock"""
    try:
        symbol = ticker.replace('.NS', '')
        ddgs = DDGS()
        results = list(ddgs.news(f"{symbol} stock India", max_results=3))
        
        headlines = []
        for item in results:
            headlines.append({
                'title': item.get('title', 'No title'),
                'source': item.get('source', 'Unknown'),
                'url': item.get('url', '#')
            })
        return headlines
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

def get_global_sentiment():
    """Check global market sentiment"""
    indices = {
        'S&P 500': '^GSPC',
        'Nikkei': '^N225',
        'Hang Seng': '^HSI',
        'Nifty 50': '^NSEI'
    }
    
    sentiment = {}
    for name, ticker in indices.items():
        try:
            data = yf.Ticker(ticker).history(period='2d')
            if len(data) >= 2:
                change = ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100
                sentiment[name] = round(change, 2)
        except:
            sentiment[name] = 0
    
    return sentiment

# ==========================================
# TECHNICAL ANALYSIS
# ==========================================

def calculate_indicators(df):
    """Calculate technical indicators"""
    # EMAs
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # MACD
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    # Volume SMA
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    # Bollinger Bands
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    return df

def screen_stock(ticker):
    """Screen individual stock"""
    df = fetch_stock_data(ticker)
    if df is None:
        return None
    
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    
    # Calculate trend strength
    ema_trend = 1 if latest['Close'] > latest['EMA_20'] > latest['EMA_50'] else 0
    volume_spike = latest['Volume'] / latest['Volume_SMA'] if latest['Volume_SMA'] > 0 else 0
    
    # Screening score
    score = 0
    reasons = []
    
    if volume_spike > 1.5:
        score += 2
        reasons.append(f"High volume ({volume_spike:.1f}x avg)")
    
    if ema_trend:
        score += 2
        reasons.append("Bullish EMA alignment")
    
    if 40 < latest['RSI'] < 70:
        score += 1
        reasons.append(f"Healthy RSI ({latest['RSI']:.1f})")
    
    if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
        score += 1
        reasons.append("MACD bullish")
    
    # Only return if score >= 4
    if score >= 4:
        return {
            'ticker': ticker,
            'price': round(latest['Close'], 2),
            'rsi': round(latest['RSI'], 2),
            'volume_ratio': round(volume_spike, 2),
            'score': score,
            'reasons': reasons,
            'change_pct': round(((latest['Close'] / df['Close'].iloc[-2]) - 1) * 100, 2)
        }
    
    return None

# ==========================================
# DEEPSEEK ANALYSIS
# ==========================================

def analyze_with_deepseek(stock, news, global_sentiment):
    """Get AI recommendation from DeepSeek"""
    
    news_text = "\n".join([f"- {n['title']}" for n in news])
    global_text = "\n".join([f"- {k}: {v:+.2f}%" for k, v in global_sentiment.items()])
    
    prompt = f"""You are a professional Indian stock trader. Analyze this stock for TODAY.

**Stock:** {stock['ticker']}
**Price:** ₹{stock['price']} (Change: {stock['change_pct']:+.2f}%)
**RSI:** {stock['rsi']}
**Volume:** {stock['volume_ratio']}x average
**Reasons for screening:** {', '.join(stock['reasons'])}

**News:**
{news_text}

**Global Markets:**
{global_text}

Provide a recommendation in EXACTLY this JSON format:
{{
    "action": "BUY" or "HOLD" or "SKIP",
    "entry": price in rupees,
    "stop_loss": price in rupees,
    "target": price in rupees,
    "risk_score": 1-10,
    "confidence": 0-100,
    "reason": "One sentence explanation"
}}

Rules:
- Only recommend BUY if confidence > 60%
- Stop loss should be 3-5% below entry
- Target should be 5-10% above entry
- Risk score: 1=safest, 10=riskiest"""

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-reasoner",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            },
            timeout=30
        )
        
        result = response.json()['choices'][0]['message']['content']
        
        # Extract JSON
        import re
        json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        
    except Exception as e:
        print(f"DeepSeek API error: {e}")
    
    return None

# ==========================================
# REPORT GENERATION
# ==========================================

def send_telegram_message(message):
    """Send message via Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    
    try:
        requests.post(url, json=data, timeout=10)
        print("Telegram message sent successfully")
    except Exception as e:
        print(f"Telegram error: {e}")

def generate_report(recommendations, global_sentiment):
    """Generate and send report"""
    
    # Filter only BUY recommendations
    buy_picks = [r for r in recommendations if r.get('action') == 'BUY']
    buy_picks.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    # Create report
    report = f"""🌅 <b>PRE-MARKET REPORT</b> 🌅
📅 {datetime.now().strftime('%d %B %Y, %A')}
⏰ Generated at {datetime.now().strftime('%I:%M %p IST')}

📊 <b>GLOBAL MARKETS</b>
{chr(10).join([f"• {k}: {v:+.2f}%" for k, v in global_sentiment.items()])}

"""
    
    if buy_picks:
        report += f"\n⭐ <b>TOP {min(3, len(buy_picks))} BUY RECOMMENDATIONS</b>\n\n"
        
        for i, stock in enumerate(buy_picks[:3], 1):
            report += f"""<b>{i}. {stock['ticker']}</b>
💰 Entry: ₹{stock['entry']:.2f}
🛑 Stop Loss: ₹{stock['stop_loss']:.2f}
🎯 Target: ₹{stock['target']:.2f}
📊 Risk: {stock['risk_score']}/10 | Confidence: {stock['confidence']}%
💡 {stock['reason']}

"""
    else:
        report += "\n⚠️ <b>NO STRONG BUY SIGNALS TODAY</b>\nConsider staying in cash or waiting for better setups.\n"
    
    report += "\n⚡ Trade wisely! This is AI analysis, not financial advice."
    
    # Send to Telegram
    send_telegram_message(report)
    
    # Save text report
    with open('latest_report.txt', 'w', encoding='utf-8') as f:
        f.write(report.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', ''))
    
    # Save JSON data for web dashboard
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime('%Y-%m-%d'),
        "time": datetime.now().strftime('%I:%M %p IST'),
        "globalSentiment": global_sentiment,
        "recommendations": buy_picks[:3],
        "totalScanned": len(NIFTY_50_TICKERS),
        "totalCandidates": len(recommendations),
        "hasBuySignals": len(buy_picks) > 0
    }
    
    with open('report_data.json', 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)
    
    print("Report generated and sent!")

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print(f"🚀 Starting pre-market scan at {datetime.now()}")
    
    # Step 1: Get global sentiment
    print("📊 Fetching global market data...")
    global_sentiment = get_global_sentiment()
    
    # Step 2: Screen stocks
    print(f"🔍 Screening {len(NIFTY_50_TICKERS)} stocks...")
    candidates = []
    
    for ticker in NIFTY_50_TICKERS:
        result = screen_stock(ticker)
        if result:
            candidates.append(result)
            print(f"✅ {ticker} passed screening (score: {result['score']})")
    
    print(f"📋 Found {len(candidates)} candidates")
    
    # Step 3: Deep analysis with AI
    recommendations = []
    
    for stock in candidates[:10]:  # Analyze top 10 to save API calls
        print(f"🤖 Analyzing {stock['ticker']} with DeepSeek...")
        news = fetch_news(stock['ticker'])
        analysis = analyze_with_deepseek(stock, news, global_sentiment)
        
        if analysis:
            analysis['ticker'] = stock['ticker']
            recommendations.append(analysis)
    
    # Step 4: Generate report
    print("📝 Generating report...")
    generate_report(recommendations, global_sentiment)
    
    print("✅ Done!")

if __name__ == "__main__":
    main()
