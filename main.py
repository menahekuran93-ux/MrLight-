import os, json, asyncio, random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import anthropic

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

ARCHETYPES = [
    {"id":"quant","name":"Quant Analyst","count":120,"prompt":"You are a quantitative analyst. Focus on data, statistics, volatility. Be precise."},
    {"id":"macro","name":"Macro Strategist","count":110,"prompt":"You are a macro strategist. Analyze rates, inflation, global flows."},
    {"id":"technical","name":"Technical Analyst","count":100,"prompt":"You are a technical analyst. Focus on charts, patterns, indicators."},
    {"id":"fundamental","name":"Fundamentals Analyst","count":100,"prompt":"You are a fundamentals analyst. Focus on earnings, valuations."},
    {"id":"sentiment","name":"Sentiment Analyst","count":100,"prompt":"You are a sentiment analyst. Focus on crowd psychology, fear/greed."},
    {"id":"geopolitical","name":"Geopolitical Strategist","count":90,"prompt":"You are a geopolitical strategist. Focus on policy, regulation, risk."},
    {"id":"options","name":"Options Trader","count":90,"prompt":"You are an options trader. Focus on flow, implied vol, positioning."},
    {"id":"retail","name":"Retail Investor","count":100,"prompt":"You are a retail investor. Think simply, driven by news and hype."},
    {"id":"institutional","name":"Institutional Investor","count":100,"prompt":"You are an institutional investor managing $50B. Think long-term."},
    {"id":"contrarian","name":"Contrarian Analyst","count":90,"prompt":"You are a contrarian. Always challenge consensus. Find the other side."},
]

class Query(BaseModel):
    question: str

async def call_archetype(a, question, context=""):
    system = a["prompt"] + "\n\nRespond ONLY with valid JSON, no markdown:\n{\"position\":\"bullish|bearish|neutral\",\"confidence\":0.0-1.0,\"argument\":\"2-3 sentences\",\"key_signal\":\"one key factor\"}"
    user = f'Market question: "{question}"'
    if context:
        user += f"\n\nOther analysts argued:\n{context}\n\nUpdate your position if convinced."
    msg = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=256, system=system, messages=[{"role":"user","content":user}])
    raw = msg.content[0].text.strip()
    try:
        data = json.loads(raw)
    except:
        data = {"position":"neutral","confidence":0.5,"argument":raw[:200],"key_signal":"unknown"}
    return {**data, "archetype":a["id"], "name":a["name"], "count":a["count"]}

def simulate(result, count):
    agents = []
    for _ in range(count):
        b = random.gauss(0, 0.15)
        c = result["confidence"] + b
        if c > 0.6: pos = result["position"]
        elif c > 0.35: pos = "neutral"
        else:
            others = [p for p in ["bullish","bearish","neutral"] if p != result["position"]]
            pos = random.choice(others)
        agents.append(pos)
    return agents

def tally(agents):
    c = {"bullish":0,"bearish":0,"neutral":0}
    for a in agents: c[a] = c.get(a,0) + 1
    return c

def sse(type, data):
    return f"data: {json.dumps({'type':type,**data})}\n\n"

async def run_swarm(question):
    yield sse("start", {"question":question})
    prev_results = None

    for rnd in range(1, 4):
        label = ["","Initial Positions","Cross-Debate","Final Convergence"][rnd]
        yield sse("round", {"round":rnd,"label":label})
        context = ""
        if prev_results:
            context = "\n".join([f"- {r['name']} ({r['position']}): {r['argument']}" for r in prev_results])

        tasks = [call_archetype(a, question, context) for a in ARCHETYPES]
        results = await asyncio.gather(*tasks)
        round_agents = []
        flips = []

        for i, (a, result) in enumerate(zip(ARCHETYPES, results)):
            agents = simulate(result, a["count"])
            round_agents.extend(agents)
            flipped = prev_results is not None and result["position"] != prev_results[i]["position"]
            if flipped:
                flips.append({"name":result["name"],"from":prev_results[i]["position"],"to":result["position"]})
            yield sse("agent", {
                "round":rnd, "name":result["name"], "archetype":result["archetype"],
                "position":result["position"], "confidence":result["confidence"],
                "argument":result["argument"], "key_signal":result["key_signal"],
                "count":a["count"], "flipped":flipped
            })

        t = tally(round_agents)
        yield sse("tally", {"round":rnd,"tally":t,"flips":flips})
        prev_results = results

    t = tally(round_agents)
    total = sum(t.values())
    bull = round(t.get("bullish",0)/total*100)
    bear = round(t.get("bearish",0)/total*100)
    neut = 100-bull-bear
    label = "Strong Buy" if bull>=65 else "Buy" if bull>=55 else "Neutral" if bull>=45 else "Sell" if bull>=35 else "Strong Sell"
    bull_args = [r["argument"] for r in prev_results if r["position"]=="bullish"][:2]
    bear_args = [r["argument"] for r in prev_results if r["position"]=="bearish"][:2]
    yield sse("verdict", {"score":bull,"label":label,"bull_pct":bull,"bear_pct":bear,"neut_pct":neut,"bull_args":bull_args,"bear_args":bear_args})

@app.post("/analyze")
async def analyze(q: Query):
    return StreamingResponse(run_swarm(q.question), media_type="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/health")
async def health():
    return {"status":"online","agents":1000}

@app.get("/")
async def root():
    return FileResponse("index.html")
