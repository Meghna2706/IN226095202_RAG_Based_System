# RAG Customer Support Assistant

A **Retrieval-Augmented Generation (RAG)** powered customer support system built with **LangGraph**, **ChromaDB**, and **HITL (Human-in-the-Loop)** escalation. This application provides intelligent, context-aware responses to customer queries using a knowledge base, with automatic escalation for complex or sensitive issues.

## What I Built

In this project, I built a customer support assistant using a RAG pipeline. The system is able to:

- Understand user queries using basic intent detection
- Retrieve relevant information from a knowledge base
- Generate answers using an LLM
- Escalate queries when confidence is low

## My Approach

I started by understanding how RAG systems work. Then I broke the problem into smaller parts:

1. Load and process documents
2. Store embeddings in a vector database
3. Retrieve relevant chunks based on query
4. Generate answers using an LLM
5. Add a simple escalation mechanism

I initially implemented a basic pipeline and later improved it by adding:
- Relevance grading
- Confidence scoring
- Human-in-the-loop escalation

## Features

✨ **Intelligent Query Processing**
- Intent classification with escalation keyword detection
- Multi-stage retrieval and relevance grading
- Confidence-based response generation
- Automatic escalation for complex issues

📚 **Knowledge Base Integration**
- Vector embeddings with Sentence Transformers
- ChromaDB vector store for fast similarity search
- Support for PDF ingestion with chunking
- Persistent storage and retrieval

🤖 **LLM Support**
- Works with Groq (free tier models like Llama 3)
- Compatible with OpenAI's GPT models
- Configurable temperature and model selection

🎯 **Human-in-the-Loop (HITL)**
- Automatic escalation for sensitive keywords
- Low-confidence response handling
- Session-based conversation tracking

🌐 **Web Interface**
- Modern, responsive chat UI
- Real-time query processing
- Desktop and mobile friendly

## Architecture

```
User Query → Intent Classifier → Retriever → Relevance Grader → Generator → Response
                ↓ (escalate)                      ↓ (no docs)        ↓ (low confidence)
              HITL Node (Escalation)
```

**Pipeline Components:**
1. **Intent Classifier**: Detects sensitive keywords and routes to escalation
2. **Retriever**: Retrieves relevant documents from ChromaDB
3. **Relevance Grader**: Filters retrieved documents using LLM
4. **Generator**: Creates final response using relevant context
5. **HITL Node**: Handles escalation to human agents

## Prerequisites

- **Python 3.9+**
- **pip** or **conda** for dependency management
- **API Key** (either Groq or OpenAI)
- **4GB+ RAM** for embeddings model

## Installation

### 1. Clone or navigate to the project directory

```bash
cd rag-project
```

### 2. Create a Python virtual environment

```bash
python -m venv rag_env
```

### 3. Activate the virtual environment

**Windows (PowerShell):**
```bash
.\rag_env\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
.\rag_env\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source rag_env/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# API Configuration
GROQ_API_KEY=your_groq_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here

# ChromaDB Configuration
CHROMA_DIR=./chroma_db
COLLECTION_NAME=customer_support_kb

# LLM Configuration
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.0

# Retrieval Configuration
TOP_K=4
CONFIDENCE_THRESHOLD=0.60
```

### Environment Variables Explained

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | Free API key from [console.groq.com](https://console.groq.com) |
| `OPENAI_API_KEY` | - | OpenAI API key (alternative to Groq) |
| `CHROMA_DIR` | `./chroma_db` | Directory for ChromaDB persistent storage |
| `COLLECTION_NAME` | `customer_support_kb` | ChromaDB collection name |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | LLM model to use |
| `LLM_TEMPERATURE` | `0.0` | Model temperature (0 = deterministic) |
| `TOP_K` | `4` | Number of documents to retrieve |
| `CONFIDENCE_THRESHOLD` | `0.60` | Threshold for escalation (0-1) |

## Usage

### Option 1: Web Interface (Recommended)

Start the FastAPI server:

```bash
python api_server.py
```

Then open your browser and navigate to:
```
http://localhost:8000
```

The web UI will be served at the root URL with a modern chat interface.

### Option 2: CLI Chat Interface

Run the interactive command-line assistant:

```bash
python rag_app.py
```

Example session:
```
========================================================
   RAG Customer Support Assistant
   Built with LangGraph + ChromaDB + HITL
========================================================
Type your question, or 'quit' to exit.

You: How do I return an item?
Assistant: Our return policy allows customers to return items within 30 days...

You: quit
Goodbye! Have a great day.
```

## Knowledge Base Setup

### Ingest PDF Documents

Before running queries, populate the ChromaDB with your knowledge base:

```bash
python ingest.py --pdf data/sample_kb/support_manual.pdf
```

### Output
```
[Ingestion] Loading PDF: data/sample_kb/support_manual.pdf
[Ingestion] Loaded X pages from PDF
[Ingestion] Created Y chunks
[Ingestion] Generating embeddings...
[Ingestion] Storing in ChromaDB...
[Ingestion] Done! Y chunks indexed in ChromaDB at './chroma_db'
```

### Multiple PDFs

To ingest multiple documents:

```bash
python ingest.py --pdf data/sample_kb/support_manual.pdf
python ingest.py --pdf data/sample_kb/faq.pdf
python ingest.py --pdf data/sample_kb/policies.pdf
```

## Project Structure

```
rag-project/
├── api_server.py              # FastAPI server with web UI
├── rag_app.py                 # CLI application entry point
├── chat_ui.html               # Web chat interface
├── ingest.py                  # PDF ingestion script
├── requirements.txt           # Python dependencies
├── .env                       # Configuration (API keys, settings)
├── README.md                  # This file
│
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── rag_graph.py       # LangGraph pipeline definition
│   ├── tools/                 # Additional tools (extensible)
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Configuration loader
│       └── document_loader.py # PDF loading and chunking
│
├── data/
│   └── sample_kb/
│       └── support_manual.md  # Sample knowledge base
│
├── chroma_db/                 # ChromaDB persistent storage
│   ├── chroma.sqlite3
│   └── [embeddings]/
│
├── rag_env/                   # Python virtual environment
└── tests/
    └── test_rag_pipeline.py   # Unit tests
```

## API Endpoints

### Chat Endpoint
**POST** `/chat`

Request body:
```json
{
  "query": "How do I track my order?",
  "session_id": "web_session"
}
```

Response:
```json
{
  "answer": "You can track your order using the tracking number sent to your email...",
  "escalated": false,
  "confidence": 0.88
}
```

### UI Endpoint
**GET** `/`

Returns the HTML chat interface.

## Escalation Triggers

Queries are automatically escalated to human agents if:

1. **Sensitive Keywords Detected**
   - Legal, lawsuit, sue, fraud, hacked, data breach, refund abuse
   
2. **No Relevant Documents Found**
   - Retriever returns documents but grader finds none relevant
   
3. **Low Confidence Response**
   - Generator confidence score below threshold (default: 0.60)

## Dependencies

Key packages:
- **langchain** (0.2.16) - LLM framework
- **langgraph** (0.2.14) - Graph-based workflow orchestration
- **chromadb** (0.5.5) - Vector database
- **sentence-transformers** (3.0.1) - Embeddings
- **fastapi** (0.112.2) - Web framework
- **langchain-groq** (0.1.9) - Groq LLM integration
- **langchain-openai** (0.1.23) - OpenAI LLM integration

See [requirements.txt](requirements.txt) for complete list.

## Configuration Examples

### Use OpenAI Instead of Groq

```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-3.5-turbo
```

### Adjust Sensitivity

Lower confidence threshold for more escalations:
```env
CONFIDENCE_THRESHOLD=0.80
```

Higher TOP_K for more context:
```env
TOP_K=8
```

## Troubleshooting

### "No API key! Add GROQ_API_KEY or OPENAI_API_KEY to .env"
- Ensure `.env` file exists in project root
- Set either `GROQ_API_KEY` or `OPENAI_API_KEY`
- Restart the server after updating `.env`

### "Internal Server Error" when querying
- Verify ChromaDB has been populated: `python ingest.py --pdf <path-to-pdf>`
- Check `.env` file for correct API key and settings
- Review server logs for detailed error messages

### Slow responses
- Reduce `TOP_K` in `.env` (fewer documents to process)
- Use faster model: `llama3-8b-8192` instead of `llama-3.3-70b-versatile`
- Add more RAM or switch from CPU to GPU for embeddings

### ChromaDB not persisting
- Ensure `CHROMA_DIR` directory is writable
- Check that `chroma.sqlite3` file exists in `CHROMA_DIR`

## Challenges Faced

- Faced issues with API errors while integrating the LLM
- Debugging the LangGraph pipeline was difficult initially
- Handling cases where no relevant documents were retrieved
- Managing environment variables and API keys

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Future Improvements

- Improve accuracy of retrieval
- Add better UI features
- Handle multi-turn conversations
- Optimize response time


---

Built with ❤️ using LangChain, LangGraph, and ChromaDB

**By: Subramani Meghna**

Intern at Innomatics Reseach Labs
