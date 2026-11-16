#!/usr/bin/env python3
"""ConsultSense: zero-shot vs few-shot sentiment classification of public comments.

Pipeline (3 small agents):
  1. Zero-shot classifier   - labels only, no examples
  2. Few-shot classifier    - same prompt + k labelled examples from a disjoint pool
  3. Error analyst          - explains *why* a comment was misclassified (categorised)
Outputs results/results.json and results/results.md (accuracy table + misclassifications).
"""
import argparse, html, json, math, os, random, re
import pandas as pd

MODEL = os.getenv("CONSULTSENSE_MODEL", "qwen/qwen3.8-27b")  # Groq model
LABEL_MAP = {0: "negative", 2: "neutral", 4: "positive"}
COLS = ["target", "id", "date", "flag", "user", "text"]
ERROR_CATEGORIES = ["sarcasm/irony", "negation", "mixed sentiment", "no clear sentiment",
                    "slang/emoji/typos", "missing context", "likely label noise", "other"]

DEFAULT_SHOTS = [
    ("The new tax rebate is a great step, finally some relief for small shops!", "positive"),
    ("This draft will hurt farmers and nobody asked us. Totally unacceptable.", "negative"),
    ("Not happy with the timeline. Wish they had consulted more people.", "negative"),
    ("Love the idea of public hearings, hope they actually happen :)", "positive"),
    ("Oh great, another committee. That will surely fix everything.", "negative"),
    ("Thanks for extending the deadline, really helpful.", "positive"),
    ("Where can I read the full text of the policy?", "neutral"),
    ("The clause on data sharing is not bad, but enforcement is vague.", "neutral"),
]

# ---------------------------------------------------------------- data
def load_sentiment140(path):
    df = pd.read_csv(path, names=COLS, encoding="latin-1")
    df["label"] = df["target"].map(LABEL_MAP)
    df["text"] = df["text"].astype(str).map(html.unescape).str.strip()
    df = df[["text", "label"]].dropna().drop_duplicates("text")
    return df

def make_splits(df, n_eval, k_shots, seed):
    labels = sorted(df.label.unique())
    ev = pd.concat([df[df.label == l].sample(n_eval // len(labels), random_state=seed) for l in labels])
    rest = df.drop(ev.index)  # few-shot pool is disjoint from the eval set (no leakage)
    shots = pd.concat([rest[rest.label == l].sample(max(1, k_shots // len(labels)), random_state=seed)
                       for l in labels]).sample(frac=1, random_state=seed)
    return ev.sample(frac=1, random_state=seed).reset_index(drop=True), list(zip(shots.text, shots.label)), labels

# ---------------------------------------------------------------- LLM
POS = {"love", "great", "good", "happy", "thanks", "awesome", "best", "nice", "yay", "excited", "win", "fun", "wonderful"}
NEG = {"hate", "bad", "sad", "worst", "awful", "terrible", "angry", "sick", "miss", "hurt", "sucks", "ugh", "unacceptable"}

def mock_label(text, labels, few):
    toks = re.findall(r"[a-z']+", text.lower())
    s = sum(t in POS for t in toks) - sum(t in NEG for t in toks)
    if few and any(t in ("not", "no", "never") or t.endswith("n't") for t in toks):
        s = -s
    if s > 0: return "positive"
    if s < 0: return "negative"
    return "neutral" if "neutral" in labels else random.choice(["positive", "negative"])

class LLM:
    """Groq-backed LLM wrapper (free tier). Reads GROQ_API_KEY from the environment."""
    def __init__(self, mock=False, pause=2.0):
        self.mock, self.pause = mock, pause  # pause = seconds between calls, to respect free-tier limits
        if not mock:
            from groq import Groq
            self.client = Groq()  # reads GROQ_API_KEY

    def ask(self, system, user, max_tokens=1500):
        import time
        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(
                    model=MODEL, max_tokens=max(max_tokens, 4000), temperature=0,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                time.sleep(self.pause)
                return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
            except Exception as e:  # rate limit / transient error -> exponential backoff
                if type(e).__name__ in ('NotFoundError', 'AuthenticationError', 'BadRequestError'):
                    raise
                wait = min(60, 4 * 2 ** attempt)
                print(f"  API error ({type(e).__name__}); retrying in {wait}s")
                time.sleep(wait)
        return ""

def parse_json(txt):
    txt = re.sub(r"```(?:json)?", "", txt)
    for o, c in (("[", "]"), ("{", "}")):
        i, j = txt.find(o), txt.rfind(c)
        if i != -1 and j > i:
            try: return json.loads(txt[i:j + 1])
            except json.JSONDecodeError: pass
    return None

def build_system(labels, shots=None):
    s = ("You are a sentiment classifier for public comments submitted to a government policy "
         f"consultation. Classify each comment's sentiment as exactly one of: {', '.join(labels)}. "
         "Judge the author's sentiment, not the topic. Output JSON only.")
    if shots:
        s += "\n\nLabelled examples:\n" + "\n".join(f"- {json.dumps(t)} -> {l}" for t, l in shots)
    return s

def classify(llm, texts, labels, shots=None, batch=10, retries=2):
    system, out = build_system(labels, shots), []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        if llm.mock:
            out += [mock_label(t, labels, bool(shots)) for t in chunk]; continue
        user = ('Classify these comments. Reply with JSON only: a list of {"id": <int>, "label": <label>}.\n\n'
                + "\n".join(f"{j + 1}. {json.dumps(t)}" for j, t in enumerate(chunk)))
        got = {}
        for _ in range(retries + 1):
            parsed = parse_json(llm.ask(system, user))
            if isinstance(parsed, list):
                got = {int(d["id"]): str(d["label"]).lower().strip() for d in parsed if "id" in d and "label" in d}
                if len(got) == len(chunk): break
        out += [got.get(j + 1, "unparsed") if got.get(j + 1) in labels else "unparsed" for j in range(len(chunk))]
    return out

def explain_error(llm, text, gold, zero, few):
    if llm.mock:
        return {"category": "other", "explanation": "(mock mode: no LLM explanation)"}
    system = ("You are an error analyst for a sentiment classifier. Given a comment, its gold label and the "
              f"model predictions, pick the best category from {ERROR_CATEGORIES} and explain in 1-2 sentences "
              'why the model went wrong (or why the gold label is questionable). JSON only: {"category":..., "explanation":...}')
    user = json.dumps({"comment": text, "gold": gold, "zero_shot": zero, "few_shot": few})
    d = parse_json(llm.ask(system, user, 400))
    return d if isinstance(d, dict) else {"category": "other", "explanation": "unparsed"}

# ---------------------------------------------------------------- metrics
def accuracy(y, p): return sum(a == b for a, b in zip(y, p)) / len(y)

def macro_f1(y, p, labels):
    f = []
    for l in labels:
        tp = sum(a == l and b == l for a, b in zip(y, p))
        fp = sum(a != l and b == l for a, b in zip(y, p))
        fn = sum(a == l and b != l for a, b in zip(y, p))
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f) / len(f)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

def paired_bootstrap(zc, fc, iters=2000, seed=0):
    rng, n = random.Random(seed), len(zc)
    d = sorted(sum(fc[i] - zc[i] for i in (rng.randrange(n) for _ in range(n))) / n for _ in range(iters))
    return d[int(.025 * iters)], d[int(.975 * iters)]

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Sentiment140 CSV (training.1600000...csv or testdata.manual...csv)")
    ap.add_argument("--n", type=int, default=200, help="eval comments (balanced)")
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--explain", type=int, default=12, help="max misclassifications to explain")
    ap.add_argument("--mock", action="store_true", help="offline heuristic 'LLM' to test the plumbing")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)

    df = load_sentiment140(a.data)
    ev, shots, labels = make_splits(df, a.n, a.shots, a.seed)
    llm = LLM(a.mock)
    print(f"Eval={len(ev)} labels={labels} shots={len(shots)} model={'MOCK' if a.mock else MODEL}")

    texts, gold = ev.text.tolist(), ev.label.tolist()
    ev["zero"] = classify(llm, texts, labels)
    ev["few"] = classify(llm, texts, labels, shots)
    zc = [int(g == p) for g, p in zip(gold, ev.zero)]
    fc = [int(g == p) for g, p in zip(gold, ev.few)]
    b = sum(z and not f for z, f in zip(zc, fc)); c = sum(f and not z for z, f in zip(zc, fc))
    n = len(ev)
    rows = {}
    for name, col, cs in (("zero-shot", "zero", zc), ("few-shot", "few", fc)):
        lo, hi = wilson(sum(cs), n)
        rows[name] = dict(accuracy=sum(cs) / n, ci95=[lo, hi], macro_f1=macro_f1(gold, ev[col].tolist(), labels),
                          unparsed=int((ev[col] == "unparsed").sum()))
    diff = rows["few-shot"]["accuracy"] - rows["zero-shot"]["accuracy"]
    dlo, dhi = paired_bootstrap(zc, fc)
    winner = "few-shot" if diff > 0 else "zero-shot" if diff < 0 else "tie"

    # error analyst: prioritise cases where examples changed the outcome, then shared failures
    ev["zc"], ev["fc"] = zc, fc
    picks = pd.concat([ev[(ev.zc == 0) & (ev.fc == 1)].head(4), ev[(ev.zc == 1) & (ev.fc == 0)].head(3),
                       ev[(ev.zc == 0) & (ev.fc == 0)].head(max(0, a.explain - 7))]).head(a.explain)
    errs = []
    for _, r in picks.iterrows():
        kind = ("fixed by few-shot" if not r.zc and r.fc else "broken by few-shot" if r.zc and not r.fc else "both wrong")
        errs.append(dict(text=r.text, gold=r.label, zero=r.zero, few=r.few, kind=kind,
                         **explain_error(llm, r.text, r.label, r.zero, r.few)))

    res = dict(model="mock" if a.mock else MODEL, n=n, labels=labels, shots=shots, metrics=rows,
               delta_accuracy=diff, delta_ci95=[dlo, dhi], mcnemar_p=mcnemar_exact(b, c),
               zero_only_correct=b, few_only_correct=c, winner=winner, errors=errs)
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=2)
    ev.drop(columns=["zc", "fc"]).to_csv(f"{a.out}/predictions.csv", index=False)

    md = [f"# ConsultSense results ({res['model']}, n={n}, {len(shots)} shots)\n",
          "| Method | Accuracy | 95% CI | Macro-F1 |", "|---|---|---|---|"]
    for k, v in rows.items():
        md.append(f"| {k} | {v['accuracy']:.1%} | {v['ci95'][0]:.1%}-{v['ci95'][1]:.1%} | {v['macro_f1']:.3f} |")
    md += [f"\n**{winner}** wins by **{abs(diff) * 100:.1f} pts** accuracy (paired-bootstrap 95% CI of the gain "
           f"[{dlo * 100:+.1f}, {dhi * 100:+.1f}] pts; McNemar exact p={res['mcnemar_p']:.3f}; "
           f"few-shot-only correct={c}, zero-shot-only correct={b}).\n", "## Example misclassifications\n",
           "| Comment | Gold | Zero | Few | Case | Category | Why |", "|---|---|---|---|---|---|---|"]
    for e in errs:
        md.append(f"| {e['text'][:140].replace('|', '/')} | {e['gold']} | {e['zero']} | {e['few']} | {e['kind']} | "
                  f"{e['category']} | {e['explanation'].replace('|', '/')} |")
    open(f"{a.out}/results.md", "w").write("\n".join(md))
    print("\n".join(md))

if __name__ == "__main__":
    main()
