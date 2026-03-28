import tkinter as tk
from tkinter import messagebox
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')


def count_words(text):
    words = word_tokenize(text)
    return len([w for w in words if w.isalnum()])


def summarize_text():
    text = input_text.get("1.0", tk.END).strip()

    if not text:
        messagebox.showwarning("Warning", "Please enter some text!")
        return

    
    input_word_count = count_words(text)

    
    sentences = sent_tokenize(text)

    if len(sentences) == 0:
        messagebox.showerror("Error", "No valid sentences found!")
        return

    
    stop_words = set(stopwords.words("english"))
    cleaned_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        words = [w for w in words if w.isalnum() and w not in stop_words]
        cleaned_sentences.append(" ".join(words))

    
    pos_filtered_sentences = []
    for sent in cleaned_sentences:
        words = word_tokenize(sent)
        tagged = pos_tag(words)

        filtered_words = [
            w for w, tag in tagged
            if tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ')
        ]

        pos_filtered_sentences.append(" ".join(filtered_words))

    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(pos_filtered_sentences)

    
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)

    
    num_sentences = max(1, int(len(sentences) * 0.4))
    num_sentences = min(num_sentences, len(sentence_scores))

    top_indices = sentence_scores.argsort()[-num_sentences:]
    top_indices = sorted(top_indices)

    summary = " ".join([sentences[i] for i in top_indices])

    
    output_word_count = count_words(summary)

    
    if input_word_count > 0:
        compression = ((input_word_count - output_word_count) / input_word_count) * 100
    else:
        compression = 0

    
    reference = " ".join(sentences[:2])
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)

    
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, summary)

    score_label.config(
        text=f"ROUGE-1: {scores['rouge1'].fmeasure:.2f} | ROUGE-L: {scores['rougeL'].fmeasure:.2f}"
    )

    wordcount_label.config(
        text=f"Input Words: {input_word_count} | Output Words: {output_word_count} | Compression: {compression:.2f}%"
    )


def save_summary():
    summary = output_text.get("1.0", tk.END).strip()

    if not summary:
        messagebox.showwarning("Warning", "No summary to save!")
        return

    with open("output.txt", "w") as file:
        file.write(summary)

    messagebox.showinfo("Saved", "Summary saved to output.txt")


root = tk.Tk()
root.title("📰 NLP News Summarizer")
root.geometry("800x650")


title = tk.Label(root, text="News Summarization System", font=("Arial", 16, "bold"))
title.pack(pady=10)


input_label = tk.Label(root, text="Enter News Article:")
input_label.pack()

input_text = tk.Text(root, height=10, width=90)
input_text.pack(pady=5)


btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

summarize_btn = tk.Button(btn_frame, text="Generate Summary", command=summarize_text, bg="blue", fg="white")
summarize_btn.grid(row=0, column=0, padx=10)

save_btn = tk.Button(btn_frame, text="Save Summary", command=save_summary, bg="green", fg="white")
save_btn.grid(row=0, column=1, padx=10)


output_label = tk.Label(root, text="Generated Summary:")
output_label.pack()

output_text = tk.Text(root, height=10, width=90)
output_text.pack(pady=5)


score_label = tk.Label(root, text="ROUGE Score:")
score_label.pack(pady=5)


wordcount_label = tk.Label(root, text="Word Count Info:")
wordcount_label.pack(pady=5)


root.mainloop()