# Real Deal Hunting Agentic AI

A (Gradio-based) agentic AI toolkit for hunting deals & predicting product prices.

This repo includes:
- A **Gradio UI** (`source/price_is_right.py`) that runs an agent pipeline to surface deal opportunities and logs activity.
- A **vector database** (`source/products_vectorstore/`) for product embeddings and visualization.
- A **memory store** (`source/memory.json`) to persist found opportunities between runs.
- A **Modal deployment** example (`source/pricer_service.py`) for hosting the price-prediction model.

---

## ✅ Quick Start (Local)

### 1) Create & activate a Python environment (Python version >= 3.11)

Using `venv`:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Configure credentials

Copy `.env` from the existing template (or create your own) and set API keys for any LLMs / services you plan to use.

At minimum, you'll typically need:
- `OPENAI_API_KEY` (OpenAI API Key)
- `PUSHOVER_USER` and `PUSHOVER_TOKEN` (if you would like to recieve Pushover notification)
- `HF_TOKEN` (Hugging Face API Key)
- Deep Neural Network file (`deep_neural_network.pth`) placed at: https://drive.google.com/drive/folders/1uq5C9edPIZ1973dArZiEO-VE13F7m8MK?usp=drive_link

> 🔒 **Security note:** Do not commit `.env` or your API keys to git.

### 4) Load the Vector Database

From the repository root:

```bash
python source/load_vector_database.py
```

### 5) Run the Gradio UI

From the repository root:

```bash
python source/price_is_right.py
```

The application will open in browser automatically. It will look for deals from internet and notify you with the best deal. The application keep running and looks for deal with an interval of 5 mins.

---

## 🧠 Core Components

### `source/price_is_right.py`
Runs the Gradio interface, displays log output, and shows the current memory of found deals.

### `source/deal_agent_framework.py`
Handles:
- persistent memory (`memory.json`)
- a Chromadb vector store (`products_vectorstore/`)
- running the agent planner to find new deals

### `source/reset_memory.py`
Clears the saved deal memory (keeps first 2 entries):

```bash
python source/reset_memory
```

---

## 🚀 Deploying the Price Predictor (Modal)

This repo includes an example Modal service at `source/pricer_service.py`.

To deploy:

```bash
modal deploy pricer_service,py
```

You must configure Modal secrets (e.g., `huggingface-secret`) and have a Modal account.

---

## Notes / Troubleshooting

- The vector store is persisted under `source/products_vectorstore/`.
- The app uses `source/memory.json` to remember deals across runs.
- If log output appears with a white background, it is styled in the Gradio HTML renderer; you can adjust the styling in `source/price_is_right.py`.
