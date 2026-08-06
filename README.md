# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July) 2026

> 🏆 This repository is my official submission for the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Mohamed Saleh Matar                  |
| Project Name     | DreamCatcher                         |
| GitHub Username  | mohamedSalehMatar                    |
| Challenge Batch  | June–July 2026                       |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en)                         |

---

# 📖 Project Overview

DreamCatch is a dream journaling system aiming at streamlining remembering, organizing and analysing your dreams!

---

# ✨ Features

* AI dream journal that takes written description of the user's dreams.
* RAG system to store, organize and retrieve dreams for analysis. 
* AI dream analyzer that answer user queries based on stored dreams and the recurring patterns between them. 

---

# 🛠️ Technologies Used

Python
PyTorch
Streamlit
LangChain
HuggingFace
FAISS
Transformers

---

# ⚙️ Installation

DreamCatcher is designed mainly for local use. The project stores dream entries locally in the database folder and can run without any cloud service dependency.

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd DreamCatcher
```

## 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 4. Start the model service

The app uses a local FastAPI service for dream entry generation. Run:

```bash
python -m uvicorn src.model_service:app --host 127.0.0.1 --port 8000
```

## 5. Launch the Streamlit app

```bash
python src/app.py
```

> The app will create and manage dream journal files locally inside the src/database folder.

---

# 🚀 Usage

1. Open the Streamlit app in your browser.
2. Use the Journal tab to enter a dream as plain text.
3. The app sends the dream to the local model service and saves the generated entry as both JSON and Markdown files.
4. Use the Preview tab to view stored dream entries.
5. Use the Analysis tab to ask questions about your saved dreams through the local RAG workflow.

---

# 📸 Demo

A local demo flow includes:

- writing a dream in the Journal tab
- saving the generated entry in the database
- previewing the stored markdown journal entry
- asking the analysis assistant questions about past dreams

---

# 📈 Results

This project demonstrates a complete local-first workflow for dream journaling, including:

- local dream entry storage
- markdown-based previewing
- AI-assisted dream analysis using a retrieval-based approach
- a simple GUI for interacting with the journal without needing an online service

---

# 🔮 Future Improvements

* Add a more polished UI and better tab navigation
* Improve the RAG prompt quality for richer dream analysis
* Add export options such as PDF or JSON backup
* Support offline model loading improvements for lower-end machines

---

# 📚 About the Challenge

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the challenge encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
