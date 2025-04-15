
# 🧠 Graph RAG – Hybrid Knowledge Graph QA System

Welcome to **Graph RAG**, a powerful and modular project that blends **Graph-based Retrieval-Augmented Generation (RAG)** with both **PDF documents** and **SQL databases**! This repo offers **two intelligent implementations** to empower your data querying and understanding using Knowledge Graphs.

---

## 🚀 Project Features

### 📘 1. PDF-Based Graph RAG (Neo4j)
- Upload any **PDF file**.
- Automatically extract content and build a **Knowledge Graph** using **Neo4j**.
- Ask natural language questions about the document.
- Get contextual answers powered by **Graph Reasoning + LLMs**.

### 🗃️ 2. SQL-Based Graph RAG
- Connect to your **SQL database**.
- Automatically generate a **graph structure** from tables, foreign keys, and relations.
- Ask questions in plain English.
- Get **auto-generated SQL queries** + answers based on your data.

---

## 🛠️ Tech Stack

- **LangChain** + **LLMs** (OpenAI / Gemini / etc.)
- **Neo4j** (Graph Database)
- **SQLAlchemy / SQLite / MySQL** (SQL Connector)
- **Streamlit** (Frontend Interface)
- **Python** 🐍
- **Graph Visualization** via HTML & NetworkX

---

## 📂 Project Structure

graph-rag/
│
├── pdf_graph_rag/        # Handles PDF uploads + Graph generation using Neo4j
├── sql_graph_rag/        # Handles SQL DB graph generation + query mapping
├── utils/                # Shared functions and helpers
├── app.py                # Main Streamlit app
├── requirements.txt
└── README.md

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

git clone https://github.com/Nishant00111/Graph-RAG.git
cd graph-rag

### 2. Create Virtual Environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

### 3. Install Requirements

pip install -r requirements.txt

### 4. Setup Neo4j (For PDF Mode)
- Install Neo4j Desktop or use Neo4j Aura.
- Update your **Neo4j URI, username, and password** in `.env` or config file.

### 5. Run the App

streamlit run app.py

---

## 💡 Use Cases

- Intelligent document Q&A for research papers, policies, or manuals.
- Natural language SQL analytics for structured databases.
- Combine unstructured (PDF) and structured (SQL) data in a smart graph.

---

## 🧠 Inspired By

- Neo4j + RAG architectures
- LangChain's GraphIndex
- Hybrid NLP + SQL systems

---

## 🤝 Contributions

PRs, issues, and feature requests are welcome! If you liked the project, don’t forget to ⭐ the repo.

---

## 📬 Contact

Made with 💙 by [Nishu](https://github.com/yNishant00111)
