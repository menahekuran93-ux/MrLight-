import json, random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import aiohttp

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def sse(t, d):
    return f"data: {json.dumps({'type': t, **d})}\n\n"

# -------- MARKET DATA --------
async def fetch_market(q):
    async with aiohttp.ClientSession() as s:
        async with s.get(f"https://query1.finance.yahoo.com/v1/finance/search?q={q}") as r:
            search = await r.json()

    if not search.get("quotes"):
        return {"error": "no ticker"}

    ticker = search["quotes"][0]["symbol"]

    async with aiohttp.ClientSession() as s:
        async with s.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1mo&interval=1d") as r:
            data = await r.json()

    closes = [c for c in data["chart"]["result"][0]["indicators"]["quote"][0]["close"] if c]
    price = closes[-1]

    gains, losses = [], []
    for i in range(1,len(closes)):
        diff = closes[i]-closes[i-1]
        gains.append(max(diff,0))
        losses.append(max(-diff,0))

    avg_gain = sum(gains[-14:])/14 if len(gains)>=14 else 0
    avg_loss = sum(losses[-14:])/14 if len(losses)>=14 else 1
    rsi = 100-(100/(1+avg_gain/avg_loss))

    return {"ticker": ticker, "price": price, "rsi": rsi}

# -------- AGENT --------
class Agent:
    def __init__(self):
        self.belief = random.uniform(0.4,0.6)
        self.capital = random.uniform(1000,10000)
        self.pos = 0
        self.risk = random.uniform(0.3,0.9)

    def update(self, market, synth, peers):
        rsi = market["rsi"]

        if rsi<30: signal=0.7
        elif rsi>70: signal=0.3
        else: signal=0.5

        peer = sum(p.belief for p in peers)/len(peers)
        trend = (synth-market["price"])/market["price"]

        self.belief = 0.4*self.belief+0.3*signal+0.2*(0.5+trend)+0.1*peer
        self.belief = max(0,min(1,self.belief))

    def act(self):
        if self.belief>0.65:
            v=self.capital*self.risk*random.uniform(0.01,0.05)
            self.pos+=v
            return v
        elif self.belief<0.35:
            v=-self.capital*self.risk*random.uniform(0.01,0.05)
            self.pos+=v
            return v
        return 0

# -------- SIMULATION --------
def simulate(market):
    agents=[Agent() for _ in range(400)]
    real=market["price"]
    synth=real
    path=[]

    for _ in range(10):
        acts=[]
        for a in agents:
            peers=random.sample(agents,5)
            a.update(market,synth,peers)
            acts.append(a.act())

        net=sum(acts)
        synth=real*(1+net/400000)

        path.append(round(synth,2))

    bull=sum(1 for a in agents if a.belief>0.6)
    bear=sum(1 for a in agents if a.belief<0.4)
    neut=len(agents)-bull-bear

    return {
        "bull":round(bull/len(agents)*100),
        "bear":round(bear/len(agents)*100),
        "neutral":round(neut/len(agents)*100),
        "synthetic":round(synth,2),
        "path":path[-5:]
    }

# -------- API --------
class Q(BaseModel):
    question:str

@app.post("/analyze")
async def analyze(q:Q):
    async def stream():
        yield sse("start",{"q":q.question})

        market=await fetch_market(q.question)

        if "error" in market:
            yield sse("error",market)
            return

        yield sse("market",market)

        result=simulate(market)

        yield sse("result",result)

    return StreamingResponse(stream(),media_type="text/event-stream")
