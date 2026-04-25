"""
api_server.py - FIXED VERSION
Removed all log suppression so you can see what's happening
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

print("=" * 50)
print("  Starting RAG Customer Support Server...")
print("=" * 50)

print("[1/3] Loading graph...")
try:
    from src.agents.rag_graph import build_graph, run_query
    graph = build_graph()
    print("[1/3] Graph loaded OK")
except Exception as e:
    print(f"[1/3] ERROR loading graph: {e}")
    sys.exit(1)

print("[2/3] Setting up FastAPI...")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
print("[2/3] FastAPI ready")

class QueryRequest(BaseModel):
    query: str
    session_id: str = "web_session"

@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("chat_ui.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h2>chat_ui.html not found in project root</h2>"

@app.post("/chat")
async def chat(req: QueryRequest):
    print(f"\n[Query] {req.query}")
    try:
        result = run_query(graph, req.query, req.session_id)
        answer = result["answer"]
        escalated = result.get("escalated", False)
        confidence = round(result.get("confidence", 1.0), 2)
        print(f"[Answer] {answer[:80]}...")
        print(f"[Escalated] {escalated} | [Confidence] {confidence}")
        return JSONResponse({
            "answer": answer,
            "escalated": escalated,
            "confidence": confidence
        })
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"[ERROR] {err}")
        return JSONResponse(
            {"answer": f"Server error: {str(e)}", "escalated": False, "confidence": 0},
            status_code=500
        )

print("[3/3] All done!")
print()
print("  Open browser at: http://localhost:8000")
print("  Press CTRL+C to stop")
print()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")