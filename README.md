# ConsultSense

Sentiment analysis of public policy-consultation comments. Zero-shot and few-shot prompting are compared;
comments where they disagree are flagged for human review.

## Files
- `consultsense.py` - benchmark pipeline (Sentiment140, zero-shot vs few-shot, error analysis)
- `server.py` - FastAPI backend (serves the UI and the API)
- `index.html` - frontend
- `.env.example` - copy to `.env` and add your Groq key

## Run the app
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env  (then edit .env)
uvicorn server:app --port 8000
```
Open http://localhost:8000. To try it without a key, set `CONSULTSENSE_MOCK=1` in `.env`.

## Benchmark tab
The tab reads `results_final/results.json`. Either copy the `results_final` folder from your Colab run
into this folder, or generate it here (needs the Sentiment140 CSV):
```bash
python consultsense.py --data path/to/training.1600000.processed.noemoticon.csv --n 100 --shots 8 --out results_final
```
