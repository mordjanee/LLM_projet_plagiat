import os
from transformers import BertTokenizer, BertModel
import torch
import faiss
import pickle
import numpy as np
from sentence_transformers import util
import chardet  # To detect file encoding if it's not UTF-8
import fitz

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# Load the tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to compute embeddings with BERT
def embed_bert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[0, 1:-1, :].mean(dim=0).numpy()
    return embeddings

# Function to check plagiarism with detailed matches
def check_plagiarism_detailed(text, tokenizer, model, index, corpus, corpus_embeddings, threshold=0.8):
    plagiarized_parts = []
    sentences = text.split(". ")  # Split the text into sentences for comparison
    for sentence in sentences:
        sentence_embedding = embed_bert(sentence, tokenizer, model)
        D, I = index.search(np.array([sentence_embedding]), k=5)
        for i in range(len(I[0])):
            similarity = util.cos_sim(sentence_embedding, corpus_embeddings[I[0][i]])
            if similarity >= threshold:
                plagiarized_parts.append({
                    'sentence': sentence,
                    'matched_text': corpus[I[0][i]],
                    'similarity': similarity.item()
                })
    return plagiarized_parts

# Paths to directories
dossier_original = "TestData1/Original/"
dossier_copie = "TestData1/Copy/"

# Load original texts and create the index
corpus = []
corpus_embeddings = []
for filename in os.listdir(dossier_original):
    if filename.endswith(".txt"):
        file_path = os.path.join(dossier_original, filename)
        encoding = detect_encoding(file_path)  # Detect encoding
        with open(file_path, 'r', encoding=encoding) as f:
            texte_original = f.read()
            corpus.append(texte_original)
            corpus_embeddings.append(embed_bert(texte_original, tokenizer, model))

corpus_embeddings = np.array(corpus_embeddings)

# Create the FAISS index
index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

# Create a file to save results
with open("resultats_plagiat.txt", "w", encoding='utf-8') as result_file:
    for filename in os.listdir(dossier_copie):
        if filename.endswith(".txt"):
            file_path = os.path.join(dossier_copie, filename)
            encoding = detect_encoding(file_path)  # Detect encoding
            with open(file_path, 'r', encoding=encoding) as f:
                texte_copie = f.read()
                resultats = check_plagiarism_detailed(
                    texte_copie, tokenizer, model, index, corpus, corpus_embeddings
                )

                if resultats:
                    result_file.write(f"Plagiat détecté dans le fichier {filename}:\n")
                    print(f"Plagiat détecté dans le fichier {filename}:")
                    for resultat in resultats:
                        result_file.write(f"  Partie plagiée: {resultat['sentence']}\n")
                        result_file.write(f"  Correspondance trouvée: {resultat['matched_text'][:100]}...\n")
                        result_file.write(f"  Similarité: {resultat['similarity']:.2f}\n\n")
                        print(f"  Partie plagiée: {resultat['sentence']}")
                        print(f"  Correspondance trouvée: {resultat['matched_text'][:100]}...")
                        print(f"  Similarité: {resultat['similarity']:.2f}")
                else:
                    result_file.write(f"Aucun plagiat détecté dans le fichier {filename}.\n")
                    print(f"Aucun plagiat détecté dans le fichier {filename}")
