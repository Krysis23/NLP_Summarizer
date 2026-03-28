# 📰 News Summarization System using NLP

## 📌 Overview
This project implements an **Extractive Text Summarization System** using Natural Language Processing (NLP).  
It takes a news article as input and generates a concise summary by selecting the most important sentences.

The system uses **TF-IDF scoring and POS tagging** to identify meaningful content and evaluate results using **ROUGE metrics**.

---

## 🎯 Features
- Automatic text summarization  
- POS-based filtering (Nouns, Verbs, Adjectives)  
- TF-IDF based sentence ranking  
- ROUGE score evaluation  
- Input & Output word count  
- Compression percentage calculation  
- Save summary to file  
- GUI using Tkinter  

---

## 🏗️ Technologies Used
- Python  
- NLTK (Natural Language Toolkit)  
- Scikit-learn  
- NumPy  
- Rouge-score  
- Tkinter (GUI)  

---

## 📂 Project Structure

NLP-News-Summarizer/
│
├── app.py              # GUI application  
├── summarizer.py       # Core NLP logic (optional)  
├── input.txt           # Sample input file  
├── output.txt          # Generated summary  
├── requirements.txt    # Dependencies  
└── README.md           # Project documentation  

---

## ⚙️ Installation

1. Clone the repository: