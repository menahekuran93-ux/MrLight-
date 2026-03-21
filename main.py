import os, json, asyncio, random, re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import aiohttp
import anthropic

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def sse(type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': type, **data})}\n\n"

def tally(agents: list) -> dict:
    c = {"bullish": 0, "bearish": 0, "neutral": 0}
    for a in agents:
        c[a] = c.get(a, 0) + 1
    return c

async def call_claude(prompt: str, user_msg: str = "", max_tokens: int = 300):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=prompt,
            messages=[{"role": "user", "content": user_msg}]
        )
    )

async def extract_ticker(question: str):
    system = """You are a financial data assistant.
Extract the single most relevant Yahoo Finance ticker symbol from the question.
Respond with ONLY the ticker symbol or NONE.
Examples: AAPL, TSLA, BTC-USD, ETH-USD, GC=F, CL=F, ^GSPC, ^IXIC, NVDA, MSFT, AMZN, GOOGL, META, AMD, COIN, SPY, QQQ
For gold use GC=F, oil use CL=F, S&P 500 use ^GSPC, Nasdaq use ^IXIC, Bitcoin use BTC-USD."""
    try:
        msg = await call_claude(system, question, max_tokens=20)
        raw = msg.content[0].text.strip().upper()
        ticker = re.findall(r'\b[A-Z0-9\^\-_=]+\b', raw)
        if not ticker or ticker[0] == "NONE":
            return None
        return ticker[0]
    except:
        return None

async def fetch_market_data(ticker: str):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1mo"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8), headers={"User-Agent": "Mozilla/5.0"}) as resp:
                data = await resp.json()
        result = data["chart"]["result"][0]
        meta = result["meta"]
        closes = [c for c in result["indicators"]["quote"][0].get("close", []) if c is not None]
        if len(closes) < 2:
            return {"ticker": ticker, "error": "Insufficient price data"}
        current = meta.get("regularMarketPrice", closes[-1])
        prev_close = meta.get("chartPreviousClose", closes[-2])
        month_change = ((closes[-1] - closes[0]) / closes[0] * 100) if closes[0] else 0
        day_change = ((current - prev_close) / prev_close * 100) if prev_close else 0
        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else (sum(gains) / len(gains) if gains else 0)
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else (sum(losses) / len(losses) if losses else 0.001)
        rsi = round(100 - (100 / (1 + avg_gain / avg_loss)), 1) if avg_loss > 0 else 50
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        avg_return = sum(returns) / len(returns) if returns else 0
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns) if returns else 0
        volatility = round((variance ** 0.5) * 100, 2)
        high_52w = meta.get("fiftyTwoWeekHigh", max(closes))
        low_52w = meta.get("fiftyTwoWeekLow", min(closes))
        return {
            "ticker": ticker,
            "current_price": round(current, 2),
            "day_change_pct": round(day_change, 2),
            "month_change_pct": round(month_change, 2),
            "52w_high": round(high_52w, 2),
            "52w_low": round(low_52w, 2),
            "pct_from_52w_high": round((current - high_52w) / high_52w * 100, 2),
            "pct_from_52w_low": round((current - low_52w) / low_52w * 100, 2),
            "rsi_14": rsi,
            "volatility_30d": volatility,
            "price_history_30d": [round(c, 2) for c in closes[-10:]]
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

async def fetch_news(question: str):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={question[:60]}&region=US&lang=en-US"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=6), headers={"User-Agent": "Mozilla/5.0"}) as resp:
                text = await resp.text()
        titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', text)
        return [t for t in titles if len(t) > 10 and "Yahoo" not in t][:6]
    except:
        return []

ARCHETYPES = [
    {
        "id": "quant",
        "name": "Quant Analyst",
        "count": 120,
        "prompt": """You are a quantitative analyst at a top hedge fund.
You rely ONLY on hard data: RSI, price momentum, distance from 52-week highs/lows, volatility.
You ignore narratives and news completely. Numbers are truth.
RSI > 70 = overbought warning. RSI < 30 = oversold opportunity.
High volatility = wider distribution of outcomes = more risk.
Always cite the specific numbers given to you."""
    },
    {
        "id": "macro",
        "name": "Macro Strategist",
        "count": 110,
        "prompt": """You are a macro strategist at a global investment bank.
You analyze how the current macro environment — interest rates, inflation, dollar strength,
global liquidity cycles — affects this specific asset.
Think in terms of risk-on vs risk-off. Where is capital flowing globally right now?
Be specific about which macro forces are dominant."""
    },
    {
        "id": "technical",
        "name": "Technical Analyst",
        "count": 100,
        "prompt": """You are a technical analyst with 20 years of experience reading charts.
Use the 30-day price history to identify: trend direction, momentum, key support/resistance.
Higher lows = uptrend intact. Lower highs = downtrend.
Distance from 52w high/low tells you where you are in the cycle.
Never mention fundamentals. Price action is everything."""
    },
    {
        "id": "fundamental",
        "name": "Fundamentals Analyst",
        "count": 100,
        "prompt": """You are a fundamental analyst obsessed with intrinsic value.
You ask: is this asset worth what it currently trades for?
For stocks: earnings quality, competitive moat, growth vs valuation.
For crypto: real utility, network effects, adoption curve.
For commodities: supply constraints, demand drivers.
Be skeptical of hype. Focus purely on value."""
    },
    {
        "id": "sentiment",
        "name": "Sentiment Analyst",
        "count": 100,
        "prompt": """You are a market sentiment analyst who reads crowd psychology.
The news headlines provided are real — use them as your primary signal.
Euphoric headlines = smart money is selling to retail. Fear headlines = accumulation opportunity.
What is the dominant emotion in the market right now for this asset?
Sentiment extremes are the most reliable contrarian signals."""
    },
    {
        "id": "risk",
        "name": "Risk Manager",
        "count": 90,
        "prompt": """You are a chief risk officer at a $10B fund. You are paid to say no.
Your job: identify every risk scenario. What could go catastrophically wrong?
Look at: drawdown from 52w high, volatility, tail risks, liquidity.
You only approve positions when risk/reward is explicitly compelling.
Default to caution. The market can always go lower."""
    },
    {
        "id": "momentum",
        "name": "Momentum Trader",
        "count": 90,
        "prompt": """You are a pure momentum trader. You have one rule: follow the trend.
Look at the month change and recent price history. Is it going up or down?
Positive momentum = bullish. Negative momentum = bearish. That simple.
You do not predict reversals. You ride trends until they break.
If momentum just reversed, that is your most important signal."""
    },
    {
        "id": "contrarian",
        "name": "Contrarian Analyst",
        "count": 90,
        "prompt": """You are a contrarian investor who profits from fading consensus.
When headlines are bullish and everyone is long — you look for reasons to be bearish.
When panic is everywhere — you see opportunity.
Use the news headlines as your sentiment gauge.
Challenge the dominant narrative with specific data."""
    },
    {
        "id": "institutional",
        "name": "Institutional Investor",
        "count": 100,
        "prompt": """You manage $50B for a pension fund. You think in 3-5 year horizons.
Short-term price swings are irrelevant noise to you.
You ask: what are the structural, secular forces driving this asset over the next 3-5 years?
You only build positions when you have deep fundamental conviction.
Current price matters only as your entry point for a long-term thesis."""
    },
    {
        "id": "retail",
        "name": "Retail Investor",
        "count": 100,
        "prompt": """You are an average retail investor who follows financial news and social media.
You are heavily influenced by recent price action and headlines.
If it went up recently you think it will keep going up (FOMO).
If headlines are scary you want to sell everything (panic).
Be honest about emotional reasoning — retail sentiment moves markets."""
    },
]

AGENT_BEHAVIORS = {
    "quant":         {"momentum_bias": 0.2,  "risk_bias": -0.05, "contrarian": False},
    "macro":         {"momentum_bias": 0.0,  "risk_bias": -0.1,  "contrarian": False},
    "technical":     {"momentum_bias": 0.3,  "risk_bias": 0.0,   "contrarian": False},
    "fundamental":   {"momentum_bias": -0.05,"risk_bias": -0.05, "contrarian": False},
    "sentiment":     {"momentum_bias": 0.05, "risk_bias": 0.0,   "contrarian": True},
    "risk":          {"momentum_bias": 0.0,  "risk_bias": -0.2,  "contrarian": False},
    "momentum":      {"momentum_bias": 0.25, "risk_bias": 0.0,   "contrarian": False},
    "contrarian":    {"momentum_bias": -0.2, "risk_bias": 0.0,   "contrarian": True},
    "institutional": {"momentum_bias": -0.05,"risk_bias": -0.05, "contrarian": False},
    "retail":        {"momentum_bias": 0.2,  "risk_bias": 0.0,   "contrarian": False},
}

def simulate_population(archetype_result: dict, count: int, market_data: dict) -> list:
    behavior = AGENT_BEHAVIORS.get(archetype_result["archetype"], {})
    base_position = archetype_result["position"]
    pos_map = {"bullish": 1.0, "neutral": 0.5, "bearish": 0.0}
    base_numeric = pos_map.get(base_position, 0.5)
    momentum_signal = max(-0.3, min(0.3, market_data.get("month_change_pct", 0) / 100 * 0.3))
    rsi = market_data.get("rsi_14", 50)
    rsi_signal = -0.1 if rsi > 70 else 0.1 if rsi < 30 else 0
    dist_high = -market_data.get("pct_from_52w_high", 0) / 100
    dist_low = market_data.get("pct_from_52w_low", 0) / 100
    volatility_noise = min(0.25, max(0.08, market_data.get("volatility_30d", 10) / 100))
    agents = []
    for _ in range(count):
        individual_noise = random.gauss(0, volatility_noise)
        archetype_momentum = behavior.get("momentum_bias", 0) * (1 + momentum_signal)
        risk_adj = behavior.get("risk_bias", 0)
        score = base_numeric + individual_noise + archetype_momentum + rsi_signal + dist_high + dist_low + risk_adj
        if behavior.get("contrarian"):
            score = 1.0 - score + random.gauss(0, 0.08)
        score = max(0.0, min(1.0, score))
        if score > 0.6: agents.append("bullish")
        elif score < 0.4: agents.append("bearish")
        else: agents.append("neutral")
    return agents

async def call_archetype(archetype: dict, question: str, market_data: dict, news: list, context: str = ""):
    market_context = ""
    if "error" not in market_data:
        market_context = f"""
REAL MARKET DATA:
- Ticker: {market_data.get('ticker')}
- Current Price: ${market_data.get('current_price')}
- Day Change: {market_data.get('day_change_pct')}%
- 30-Day Change: {market_data.get('month_change_pct')}%
- 52W High: ${market_data.get('52w_high')} ({market_data.get('pct_from_52w_high')}% from high)
- 52W Low: ${market_data.get('52w_low')} ({market_data.get('pct_from_52w_low')}% from low)
- RSI(14): {market_data.get('rsi_14')}
- Volatility(30d): {market_data.get('volatility_30d')}%
- Recent Prices: {market_data.get('price_history_30d')}"""
    news_context = "\nNEWS:\n" + "\n".join(f"- {h}" for h in news) if news else ""
    debate_context = f"\nDEBATE SO FAR:\n{context}\nRespond to these arguments. Change position if the data compels you." if context else ""
    prompt = f"""{archetype['prompt']}
Analyze using real market data provided. Cite specific numbers.
Respond ONLY in JSON: {{"position":"bullish|bearish|neutral","confidence":0.0-1.0,"argument":"2-3 sentences citing data","key_signal":"single most important factor"}}"""
    user_msg = f"Question: {question}\n{market_context}{news_context}{debate_context}"
    try:
        raw = await call_claude(prompt, user_msg, 300)
        raw_text = raw.content[0].text.strip()
        try:
            data = json.loads(raw_text)
        except:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            data = json.loads(match.group()) if match else {"position": "neutral", "confidence": 0.5, "argument": raw_text[:200], "key_signal": "parse error"}
        return {**data, "archetype": archetype["id"], "name": archetype["name"], "count": archetype["count"]}
    except Exception as e:
        return {"position": "neutral", "confidence": 0.5, "argument": str(e), "key_signal": "error", "archetype": archetype["id"], "name": archetype["name"], "count": archetype["count"]}

async def run_swarm(question: str):
    yield sse("start", {"question": question})
    ticker = await extract_ticker(question)
    market_data = await fetch_market_data(ticker) if ticker else {"error": "No asset identified"}
    news = await fetch_news(question)
    yield sse("market_data", {"data": market_data, "news": news, "has_data": "error" not in market_data})
    prev_results = None
    all_agents = []
    for rnd in range(1, 4):
        label = ["", "Initial Positions", "Cross-Debate", "Final Convergence"][rnd]
        yield sse("round", {"round": rnd, "label": label})
        context = "\n".join([f"- {r['name']} ({r['position']}, {int(r['confidence']*100)}%): {r['argument']}" for r in prev_results]) if prev_results else ""
        tasks = [call_archetype(a, question, market_data, news, context) for a in ARCHETYPES]
        results = await asyncio.gather(*tasks)
        round_agents = []
        flips = []
        for i, (archetype, result) in enumerate(zip(ARCHETYPES, results)):
            agents = simulate_population(result, archetype["count"], market_data)
            round_agents.extend(agents)
            flipped = prev_results is not None and result["position"] != prev_results[i]["position"]
            if flipped:
                flips.append({"name": result["name"], "from": prev_results[i]["position"], "to": result["position"]})
            yield sse("agent", {**result, "round": rnd, "flipped": flipped})
        t = tally(round_agents)
        yield sse("tally", {"round": rnd, "tally": t, "flips": flips})
        prev_results = results
        all_agents = round_agents
    t = tally(all_agents)
    total = sum(t.values())
    bull = round(t.get("bullish", 0) / total * 100)
    bear = round(t.get("bearish", 0) / total * 100)
    neut = 100 - bull - bear
    label = "Strong Buy" if bull >= 65 else "Buy" if bull >= 55 else "Neutral" if bull >= 45 else "Sell" if bull >= 35 else "Strong Sell"
    bull_args = [r["argument"] for r in prev_results if r["position"] == "bullish"][:3]
    bear_args = [r["argument"] for r in prev_results if r["position"] == "bearish"][:3]
    yield sse("verdict", {"score": bull, "label": label, "bull_pct": bull, "bear_pct": bear, "neut_pct": neut, "bull_args": bull_args, "bear_args": bear_args, "ticker": market_data.get("ticker", ""), "current_price": market_data.get("current_price", ""), "news": news})

class Query(BaseModel):
    question: str

@app.post("/analyze")
async def analyze(q: Query):
    return StreamingResponse(run_swarm(q.question), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/health")
async def health():
    return {"status": "online", "agents": 1000, "archetypes": len(ARCHETYPES)}

@app.get("/")
async def root():
    return FileResponse("index.html")
