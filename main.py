import os, json, asyncio, random, re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import aiohttp

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ----------------- SSE Helper -----------------
def sse(type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': type, **data})}\n\n"

def tally(agents: list) -> dict:
    c = {"bullish": 0, "bearish": 0, "neutral": 0}
    for a in agents:
        c[a] = c.get(a, 0) + 1
    return c

# ----------------- Ticker Extraction via Yahoo -----------------
async def extract_ticker(question: str):
    query = question.strip()
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=6) as resp:
                data = await resp.json()
        if data.get("quotes"):
            return data["quotes"][0]["symbol"]
    except:
        pass
    return None

# ----------------- Market Data -----------------
async def fetch_market_data(ticker: str):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1mo"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=8) as resp:
                data = await resp.json()
        result = data["chart"]["result"][0]
        meta = result["meta"]
        closes = [c for c in result["indicators"]["quote"][0].get("close", []) if c is not None]
        if len(closes)<2:
            return {"ticker": ticker, "error": "Insufficient price data"}
        current = meta.get("regularMarketPrice", closes[-1])
        prev_close = meta.get("chartPreviousClose", closes[-2])
        month_change_pct = ((closes[-1]-closes[0])/closes[0]*100) if closes[0] else 0
        gains, losses = [], []
        for i in range(1,len(closes)):
            diff = closes[i]-closes[i-1]
            gains.append(max(diff,0))
            losses.append(max(-diff,0))
        avg_gain = sum(gains[-14:])/14 if len(gains)>=14 else (sum(gains)/len(gains) if gains else 0)
        avg_loss = sum(losses[-14:])/14 if len(losses)>=14 else (sum(losses)/len(losses) if losses else 0.001)
        rsi_14 = round(100-(100/(1+avg_gain/avg_loss)),1) if avg_loss>0 else 50
        returns = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
        avg_return = sum(returns)/len(returns) if returns else 0
        variance = sum((r-avg_return)**2 for r in returns)/len(returns) if returns else 0
        volatility_30d = round((variance**0.5)*100,2)
        high_52w = meta.get("fiftyTwoWeekHigh", max(closes))
        low_52w = meta.get("fiftyTwoWeekLow", min(closes))
        return {
            "ticker": ticker,
            "current_price": round(current,2),
            "month_change_pct": round(month_change_pct,2),
            "52w_high": round(high_52w,2),
            "52w_low": round(low_52w,2),
            "pct_from_52w_high": round((current-high_52w)/high_52w*100,2),
            "pct_from_52w_low": round((current-low_52w)/low_52w*100,2),
            "rsi_14": rsi_14,
            "volatility_30d": volatility_30d,
            "price_history_30d": [round(c,2) for c in closes[-10:]],
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ----------------- News -----------------
async def fetch_news(question: str):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={question[:60]}&region=US&lang=en-US"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=6) as resp:
                text = await resp.text()
        titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', text)
        return [t for t in titles if len(t)>10 and "Yahoo" not in t][:6]
    except:
        return []

# ----------------- Archetypes & Behaviors -----------------
ARCHETYPES = [
    {"id":"quant","name":"Quant Analyst","count":120},
    {"id":"technical","name":"Technical Analyst","count":100},
    {"id":"momentum","name":"Momentum Trader","count":90},
    {"id":"risk","name":"Risk Manager","count":90},
    {"id":"sentiment","name":"Sentiment Analyst","count":100},
    {"id":"contrarian","name":"Contrarian Analyst","count":90},
    {"id":"fundamental","name":"Fundamental Analyst","count":100},
    {"id":"macro","name":"Macro Strategist","count":110},
    {"id":"institutional","name":"Institutional Investor","count":100},
    {"id":"retail","name":"Retail Investor","count":100},
]

AGENT_BEHAVIORS = {
    "quant":{"primary":"rsi_14","weight":0.4,"noise":0.03},
    "technical":{"primary":"price_history_30d","weight":0.5,"noise":0.03},
    "momentum":{"primary":"month_change_pct","weight":0.6,"noise":0.03},
    "risk":{"primary":"volatility_30d","weight":-0.5,"noise":0.03},
    "sentiment":{"primary":"news","weight":0.4,"noise":0.03},
    "contrarian":{"primary":"contrarian","weight":-0.4,"noise":0.03},
    "fundamental":{"primary":"fundamental","weight":0.5,"noise":0.03},
    "macro":{"primary":"macro","weight":0.3,"noise":0.03},
    "institutional":{"primary":"structural","weight":0.2,"noise":0.03},
    "retail":{"primary":"momentum","weight":0.25,"noise":0.03},
}

# ----------------- Enhanced Agent Simulation -----------------
def simulate_population(archetype_id: str, count: int, market_data: dict, news: list):
    agents=[]
    behavior=AGENT_BEHAVIORS.get(archetype_id)
    for _ in range(count):
        score=0.5
        primary=behavior["primary"]
        w=behavior["weight"]
        # --- primary signal handling ---
        if primary=="rsi_14" and "rsi_14" in market_data:
            rsi=market_data["rsi_14"]
            if rsi>70: score-=w
            elif rsi<30: score+=w
        elif primary=="month_change_pct" and "month_change_pct" in market_data:
            pct=market_data["month_change_pct"]
            if pct>5: score+=w
            elif pct<-5: score-=w
            else: score+=w*(pct/5)
        elif primary=="price_history_30d" and "price_history_30d" in market_data:
            trend=(market_data["price_history_30d"][-1]-market_data["price_history_30d"][0])/market_data["price_history_30d"][0]
            score+=w*trend
        elif primary=="volatility_30d" and "volatility_30d" in market_data:
            score+=w*(-market_data["volatility_30d"]/50)
        elif primary=="news":
            sentiment=0
            for h in news:
                h_lower=h.lower()
                if any(x in h_lower for x in ["sell","panic","drop"]): sentiment-=0.3
                elif any(x in h_lower for x in ["buy","rally","gain"]): sentiment+=0.3
            score+=w*sentiment
        elif primary=="contrarian" or archetype_id=="contrarian":
            score=1.0-score
        # --- secondary signals ---
        if primary!="rsi_14" and "rsi_14" in market_data:
            rsi=market_data["rsi_14"]
            if rsi>75: score-=0.1
            elif rsi<25: score+=0.1
        if primary!="month_change_pct" and "month_change_pct" in market_data:
            score+=0.05*(market_data["month_change_pct"]/5)
        # --- noise ---
        score+=random.gauss(0,behavior["noise"])
        score=max(0.0,min(1.0,score))
        # --- map to position ---
        if score>0.6: agents.append("bullish")
        elif score<0.4: agents.append("bearish")
        else: agents.append("neutral")
    return agents

# ----------------- Swarm Runner -----------------
async def run_swarm(question: str):
    yield sse("start",{"question":question})
    ticker=await extract_ticker(question)
    market_data=await fetch_market_data(ticker) if ticker else {"error":"No asset identified"}
    news=await fetch_news(question)
    yield sse("market_data",{"data":market_data,"news":news,"has_data":"error" not in market_data})

    all_agents=[]
    for rnd in range(1,4):
        yield sse("round",{"round":rnd,"label":["","Initial Positions","Cross-Debate","Final Convergence"][rnd]})
        for arch in ARCHETYPES:
            agents=simulate_population(arch["id"],arch["count"],market_data,news)
            all_agents.extend(agents)
            counts=tally(agents)
            yield sse("agent",{"archetype":arch["id"],"counts":counts,"round":rnd})
    final=tally(all_agents)
    total=sum(final.values())
    bull=round(final.get("bullish",0)/total*100)
    bear=round(final.get("bearish",0)/total*100)
    neut=100-bull-bear
    label="Strong Buy" if bull>=65 else "Buy" if bull>=55 else "Neutral" if bull>=45 else "Sell" if bull>=35 else "Strong Sell"
    yield sse("verdict",{"score":bull,"label":label,"bull_pct":bull,"bear_pct":bear,"neut_pct":neut,"ticker":market_data.get("ticker",""),"current_price":market_data.get("current_price",""),"news":news})

# ----------------- Routes -----------------
class Query(BaseModel):
    question:str

@app.post("/analyze")
async def analyze(q: Query):
    return StreamingResponse(run_swarm(q.question),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/health")
async def health():
    return {"status":"online","agents":sum(a['count'] for a in ARCHETYPES),"archetypes":len(ARCHETYPES)}

@app.get("/")
async def root():
    return FileResponse("index.html")
