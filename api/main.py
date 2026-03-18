from fastapi import FastAPI
from rag.pipeline import run_pipeline

app = FastAPI()

@app.post("/ask")
def ask(question: str):
    answer = run_pipeline(question)
    return {"answer": answer}