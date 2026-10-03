# 🤖 AI PDF Question Answering System

An AI-powered PDF Question Answering System that allows users to upload a PDF document and ask questions about its content.

The system extracts text from the uploaded PDF, processes the content, retrieves the most relevant information using vector-based retrieval, and generates an answer based on the document.

---

## 📌 Project Overview

Reading and searching through large PDF documents manually can be time-consuming.

This project provides an intelligent solution where users can:

- 📄 Upload a PDF document
- 🔍 Ask questions related to the PDF
- 🧠 Retrieve relevant information from the document
- 💬 Generate answers based on the retrieved content
- ⚡ Get answers without manually searching through the entire document

The system follows a **Retrieval-Augmented Generation (RAG)** approach.

---

## 🏗️ System Architecture

```text
                 ┌──────────────────┐
                 │    PDF Upload    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   PDF Text       │
                 │   Extraction     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Text Chunking    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Text Embeddings  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Vector Retrieval │
                 └────────┬─────────┘
                          │
                  User Question
                          │
                          ▼
                 ┌──────────────────┐
                 │ Relevant Context │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   AI / LLM       │
                 │ Answer Generation│
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   Final Answer   │
                 └──────────────────┘
