"""ConsultSense backend (FastAPI). Run: uvicorn server:app --port 8000   (set CONSULTSENSE_MOCK=1 to test without an API key)"""
import json, os
try:  # optional: load GROQ_API_KEY etc. from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from consultsense import LLM, classify, parse_json, DEFAULT_SHOTS

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK = os.getenv("CONSULTSENSE_MOCK") == "1"
LABELS = ["positive", "negative", "neutral"]
MAX_COMMENTS = 60
app = FastAPI(title="ConsultSense")
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        try:
            _llm = LLM(MOCK)
        except Exception as e:
            raise HTTPException(500, f"Could not start the model client: {e}. Check that GROQ_API_KEY is set.")
    return _llm

class AnalyzeRequest(BaseModel):
    policy: str = ""
    comments: list[str]

def summarise_themes(llm, policy, rows):
    """Agent 3: turns labelled comments into a short brief for the ministry."""
    if llm.mock:
        return {"headline": "Mock mode: no theme summary.", "supportive": [], "concerns": [], "questions": []}
    system = ("You brief a government ministry on public feedback. Given comments with sentiment labels, reply with JSON only: "
              '{"headline": one sentence, "supportive": [up to 3 short themes], "concerns": [up to 3 short themes], '
              '"questions": [up to 3 questions people are asking]}. Use only what the comments say.')
    user = json.dumps({"policy": policy or "unspecified", "comments": [{"text": r["comment"], "sentiment": r["sentiment"]} for r in rows]})
    try:
        d = parse_json(llm.ask(system, user, 800))
        return d if isinstance(d, dict) else None
    except Exception:
        return None

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    comments = [c.strip() for c in req.comments if c and c.strip()]
    if not comments:
        raise HTTPException(400, "Add at least one comment.")
    if len(comments) > MAX_COMMENTS:
        raise HTTPException(400, f"Please send at most {MAX_COMMENTS} comments at a time (you sent {len(comments)}).")
    llm = get_llm()
    try:
        zero = classify(llm, comments, LABELS)
        few = classify(llm, comments, LABELS, DEFAULT_SHOTS)
    except Exception as e:
        raise HTTPException(502, f"The model call failed ({type(e).__name__}). Check the model name and API key.")
    rows = [{"comment": c, "zero_shot": z, "few_shot": f, "sentiment": f if f != "unparsed" else z,
             "needs_review": z != f} for c, z, f in zip(comments, zero, few)]
    counts = {l: sum(r["sentiment"] == l for r in rows) for l in LABELS}
    return {"policy": req.policy, "total": len(rows), "counts": counts,
            "needs_review": sum(r["needs_review"] for r in rows), "rows": rows,
            "themes": summarise_themes(llm, req.policy, rows)}

@app.get("/api/benchmark")
def benchmark():
    for d in ("results_final", "results_main", "results"):
        p = os.path.join(HERE, d, "results.json")
        if os.path.exists(p):
            return json.load(open(p))
    raise HTTPException(404, "No benchmark results yet. Run consultsense.py first.")

@app.get("/api/health")
def health():
    return {"ok": True, "mock": MOCK, "model": os.getenv("CONSULTSENSE_MODEL", "qwen/qwen3.8-27b")}

@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))
