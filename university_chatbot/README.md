# University Chatbot with LLM-as-a-Judge

This project implements an intelligent chatbot for a university, capable of answering student inquiries by utilizing a Retrieval-Augmented Generation (RAG) pipeline. Crucially, it incorporates an "LLM-as-a-Judge" component that automatically evaluates the quality, correctness, and safety of the chatbot's responses.

## Architecture

```
User Query -> [ Retriever ] -> Context -> [ Generator LLM ] -> Answer
                                                                 |
                                                                 v
                                                        [ Judge LLM ] -> (Evaluation Score, Hallucination Check) -> Database
```

## Setup Instructions

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   DATABASE_URL=sqlite:///data/chatbot.db
   ```

## Usage

### Ingesting Documents
To populate the knowledge base, place your documents in the `knowledge_base` subdirectories (admissions, academic, etc.) and run the ingestion script (assuming an `ingest.py` exists):
```bash
python scripts/ingest.py
```

### Running the API Server
Start the backend server to interact with the chatbot:
```bash
python app.py
```

### Running Batch Evaluation
To evaluate the chatbot against the dataset, run:
```bash
python -m evaluation.run_evaluation --dataset evaluation/dataset.json --output-dir evaluation/results --threshold 0.7
```

## Project Structure Overview

- `database/`: Contains SQLAlchemy ORM models for logging conversations and evaluation metrics.
- `evaluation/`: Scripts and datasets for batch testing the pipeline.
- `knowledge_base/`: Directory to store source documents categorized by topic.
- `logs/`: Application logs.
- `data/`: Local storage, such as SQLite database files.

## Technology Stack

- **Python**: Primary programming language.
- **LangChain**: Framework for building the LLM pipeline.
- **SQLAlchemy 2.0**: ORM for database management.
- **Rich**: Terminal output formatting for evaluations.
- **OpenAI API**: For both generation and evaluation (LLM-as-a-Judge).

## How the Judge Works

The LLM-as-a-Judge evaluates generated answers across several metrics:
- **Faithfulness**: Does the answer rely solely on the retrieved context?
- **Correctness**: Is the answer factually correct based on the context and expected answer?
- **Relevance**: Does the answer address the user's specific query?
- **Completeness**: Is the answer comprehensive?
- **Clarity**: Is the answer easy to understand?

The judge provides an overall score and flags any hallucinations. If the score falls below a threshold, the response fails the evaluation, helping to ensure high-quality and safe interactions for users.
