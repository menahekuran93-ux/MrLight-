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

# ----------------- Helpers -----------------
def sse(type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': type, **data})}\n\n"

def tally(agents: list) -> dict:
    c = {"bullish": 0, "bearish": 0, "neutral": 0}
    for a in agents:
        c[a] = c.get(a, 0) + 1
    return c

# ----------------- Async Claude -----------------
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

# ----------------- Ticker Extraction -----------------
async def extract_ticker(question: str):
    system = """You are a financial data assistant.
Extract the single most relevant Yahoo Finance ticker symbol from the question.
Respond with ONLY the ticker symbol or NONE."""
    try:
        msg = await call_claude(system, question, max_tokens=20)
        raw = msg.content[0].text.strip().upper()
        ticker = re.findall(r'\b[A-Z0-9\^\-_=]+\b', raw)
        if not ticker or ticker[0] == "NONE":
            return None
        return ticker[0]
    except:
        return None

# ----------------- Market Data -----------------
async def fetch_market_data(ticker: str):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1mo"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                data = await resp.json()
        result = data["chart"]["result"][0]
        meta = result["meta"]
        closes = [c for c in result["indicators"]["quote"][0].get("close", []) if c is not None]
        if len(closes) < 2:
            return {"ticker": ticker, "error": "Insufficient price data"}
        current = meta.get("regularMarketPrice", closes[-1])
        prev_close = meta.get("chartPreviousClose", closes[-2])
        month_change = ((closes[-1]-closes[0])/closes[0]*100) if closes[0] else 0
        gains, losses = [], []
        for i in range(1,len(closes)):
            diff = closes[i]-closes[i-1]
            gains.append(max(diff,0))
            losses.append(max(-diff,0))
        avg_gain = sum(gains[-14:])/14 if len(gains)>=14 else (sum(gains)/len(gains) if gains else 0)
        avg_loss = sum(losses[-14:])/14 if len(losses)>=14 else (sum(losses)/len(losses) if losses else 0.001)
        rsi = round(100 - (100/(1+avg_gain/avg_loss)),1) if avg_loss>0 else 50
        returns = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
        avg_return = sum(returns)/len(returns) if returns else 0
        variance = sum((r-avg_return)**2 for r in returns)/len(returns) if returns else 0
        volatility = round((variance**0.5)*100,2)
        high_52w = meta.get("fiftyTwoWeekHigh", max(closes))
        low_52w = meta.get("fiftyTwoWeekLow", min(closes))
        return {
            "ticker": ticker,
            "current_price": round(current,2),
            "month_change_pct": round(month_change,2),
            "52w_high": round(high_52w,2),
            "52w_low": round(low_52w,2),
            "pct_from_52w_high": round((current-high_52w)/high_52w*100,2),
            "pct_from_52w_low": round((current-low_52w)/low_52w*100,2),
            "rsi_14": rsi,
            "volatility_30d": volatility,
            "price_history_30d": [round(c,2) for c in closes[-10:]]
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ----------------- News -----------------
async def fetch_news(question: str):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={question[:60]}&region=US&lang=en-US"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                text = await resp.text()
        titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', text)
        return [t for t in titles if len(t)>10 and "Yahoo" not in t][:6]
    except:
        return []

# ----------------- Archetypes -----------------
ARCHETYPES = [
    # Same archetypes as your original list
    # ...
]

AGENT_BEHAVIORS = {
    "quant":         {"momentum_bias": 0.2, "risk_bias": -0.05, "contrarian": False},
    "macro":         {"momentum_bias": 0.0, "risk_bias": -0.1, "contrarian": False},
    "technical":     {"momentum_bias": 0.3, "risk_bias": 0.0, "contrarian": False},
    "fundamental":   {"momentum_bias": -0.05,"risk_bias": -0.05,"contrarian": False},
    "sentiment":     {"momentum_bias": 0.05,"risk_bias": 0.0,"contrarian": True},
    "risk":          {"momentum_bias": 0.0,"risk_bias": -0.2,"contrarian": False},
    "momentum":      {"momentum_bias": 0.25,"risk_bias": 0.0,"contrarian": False},
    "contrarian":    {"momentum_bias": -0.2,"risk_bias": 0.0,"contrarian": True},
    "institutional": {"momentum_bias": -0.05,"risk_bias": -0.05,"contrarian": False},
    "retail":        {"momentum_bias": 0.2,"risk_bias": 0.0,"contrarian": False},
}

# ----------------- Professional Simulation -----------------
def simulate_population(archetype_result: dict, count: int, market_data: dict) -> list:
    behavior = AGENT_BEHAVIORS.get(archetype_result["archetype"], {})
    base_position = archetype_result["position"]
    pos_map = {"bullish":1.0,"neutral":0.5,"bearish":0.0}
    base_numeric = pos_map.get(base_position,0.5)

    # Signals
    momentum_signal = market_data.get("month_change_pct",0)/100*0.3
    rsi = market_data.get("rsi_14",50)
    rsi_signal = -0.1 if rsi>70 else 0.1 if rsi<30 else 0
    dist_high = -market_data.get("pct_from_52w_high",0)/100
    dist_low = market_data.get("pct_from_52w_low",0)/100
    volatility_noise = min(0.25,max(0.08,market_data.get("volatility_30d",10)/100))

    agents=[]
    for _ in range(count):
        individual_noise=random.gauss(0,volatility_noise)
        archetype_momentum = behavior.get("momentum_bias",0)*(1+momentum_signal)
        risk_adj = behavior.get("risk_bias",0)

        # Combine signals
        score=base_numeric+individual_noise+archetype_momentum+rsi_signal+dist_high+dist_low+risk_adj

        # Contrarian flip
        if behavior.get("contrarian"):
            score=1.0-score+random.gauss(0,0.08)
        score=max(0.0,min(1.0,score))

        if score>0.6: agents.append("bullish")
        elif score<0.4: agents.append("bearish")
        else: agents.append("neutral")
    return agents

# ----------------- Archetype AI -----------------
async def call_archetype(archetype: dict, question: str, market_data: dict, news: list, context: str=""):
    market_context = ""
    if "error" not in market_data:
        market_context = f"REAL MARKET DATA:\nTicker: {market_data.get('ticker')}\nPrice: {market_data.get('current_price')}\nChange30d: {market_data.get('month_change_pct')}\nRSI: {market_data.get('rsi_14')}\nVol30d: {market_data.get('volatility_30d')}\n52wHigh/Low: {market_data.get('52w_high')}/{market_data.get('52w_low')}\nRecent prices: {market_data.get('price_history_30d')}"
    news_context=""
    if news: news_context="\nNews:\n"+"\n".join(f"- {h}" for h in news)
    debate_context=f"\nDebate context:\n{context}" if context else ""

    prompt=f"""{archetype['prompt']}
Analyze the question using the market data and news provided.
Respond ONLY in JSON: {{"position":"bullish|bearish|neutral","confidence":0.0-1.0,"argument":"2-3 sentences","key_signal":"single factor"}}"""
    user_msg=f"Question: {question}\n{market_context}\n{news_context}\n{debate_context}"

    try:
        raw = await call_claude(prompt,user_msg,300)
        raw_text = raw.content[0].text.strip()
        try: data=json.loads(raw_text)
        except:
            match=re.search(r'\{.*\}',raw_text,re.DOTALL)
            data=json.loads(match.group()) if match else {"position":"neutral","confidence":0.5,"argument":raw_text[:200],"key_signal":"parse error"}
        return {**data,"archetype":archetype["id"],"name":archetype["name"],"count":archetype["count"]}
    except Exception as e:
        return {"position":"neutral","confidence":0.5,"argument":str(e),"key_signal":"error","archetype":archetype["id"],"name":archetype["name"],"count":archetype["count"]}

# ----------------- Swarm Runner -----------------
async def run_swarm(question: str):
    yield sse("start",{"question":question})
    ticker = await extract_ticker(question)
    market_data = await fetch_market_data(ticker) if ticker else {"error":"No asset identified"}
    news = await fetch_news(question)
    yield sse("market_data",{"data":market_data,"news":news,"has_data":"error" not in market_data})

    prev_results=None
    all_agents=[]
    for rnd in range(1,4):
        label=["","Initial Positions","Cross-Debate","Final Convergence"][rnd]
        yield sse("round",{"round":rnd,"label":label})
        context="\n".join([f"- {r['name']} ({r['position']}): {r['argument']}" for r in prev_results]) if prev_results else ""
        tasks=[call_archetype(a,question,market_data,news,context) for a in ARCHETYPES]
        results=await asyncio.gather(*tasks)

        round_agents=[]
        for i,(archetype,result) in enumerate(zip(ARCHETYPES,results)):
            agents=simulate_population(result,archetype["count"],market_data)
            round_agents.extend(agents)
            flipped=prev_results is not None and result["position"]!=prev_results[i]["position"]
            yield sse("agent",{**result,"round":rnd,"flipped":flipped})
        t=tally(round_agents)
        yield sse("tally",{"round":rnd,"tally":t})
        prev_results=results
        all_agents=round_agents

    # Final verdict
    t=tally(all_agents)
    total=sum(t.values())
    bull=round(t.get("bullish",0)/total*100)
    bear=round(t.get("bearish",0)/total*100)
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
    return {"status":"online","agents":1000,"archetypes":len(ARCHETYPES)}

@app.get("/")
async def root():
    return FileResponse("index.html")
