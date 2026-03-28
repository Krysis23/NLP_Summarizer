

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger_eng')

with open("input.txt", "r") as file:
    text = file.read()


sentences = sent_tokenize(text)

if len(sentences) == 0:
    print("❌ No sentences found!")
    exit()


stop_words = set(stopwords.words("english"))
cleaned_sentences = []

for sentence in sentences:
    words = word_tokenize(sentence.lower())

    words = [
        w for w in words
        if w.isalnum() and w not in stop_words
    ]

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


if len(pos_filtered_sentences) == 0:
    print("❌ No valid sentences after processing!")
    exit()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(pos_filtered_sentences)

sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)


num_sentences = max(1, int(len(sentences) * 0.4))
num_sentences = min(num_sentences, len(sentence_scores))

top_indices = sentence_scores.argsort()[-num_sentences:]
top_indices = sorted(top_indices)

summary = " ".join([sentences[i] for i in top_indices])

words = word_tokenize(text.lower())
bigrams = list(ngrams(words, 2))


reference = " ".join(sentences[:2])

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference, summary)

print("\n===== ORIGINAL TEXT =====\n")
print(text)

print("\n===== GENERATED SUMMARY =====\n")
print(summary)

print("\n===== ROUGE SCORES =====\n")
print(scores)

print("\n===== SAMPLE BIGRAMS =====\n")
print(bigrams[:10])


with open("output.txt", "w") as file:
    file.write(summary)

print("\n✅ Summary saved to output.txt")