import os
from transformers import BertTokenizer, BertModel
import torch
import faiss
import pickle
import numpy as np
from sentence_transformers import util

# Charger le tokenizer et le modèle BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embed_bert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[0, 1:-1, :].mean(dim=0).numpy()
    return embeddings

def check_plagiarism(text, tokenizer, model, index, corpus_embeddings, threshold=0.8):
    text_embedding = embed_bert(text, tokenizer, model)
    D, I = index.search(np.array([text_embedding]), k=5)
    similar_texts = []
    for i in range(len(I[0])):
        similarity = util.cos_sim(text_embedding, corpus_embeddings[I[0][i]])
        if similarity >= threshold:
            similar_texts.append({
                'text': corpus[I[0][i]],
                'similarity': similarity.item()
            })
    return similar_texts

# Chemins des dossiers
dossier_original = "TestData1/Original/"
dossier_copie = "TestData1/Copy/"

# Charger les textes originaux et créer l'index
corpus = []
corpus_embeddings = []
for filename in os.listdir(dossier_original):
    if filename.endswith(".txt"):
        with open(os.path.join(dossier_original, filename), 'r') as f:
            texte_original = f.read()
            corpus.append(texte_original)
            corpus_embeddings.append(embed_bert(texte_original, tokenizer, model))

corpus_embeddings = np.array(corpus_embeddings)

# Créer l'index FAISS
index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

# Vérifier le plagiat
for filename in os.listdir(dossier_copie):
    if filename.endswith(".txt"):
        with open(os.path.join(dossier_copie, filename), 'r') as f:
            texte_copie = f.read()
            resultats = check_plagiarism(texte_copie, tokenizer, model, index, corpus_embeddings)

            if resultats:
                print(f"Plagiat détecté dans le fichier {filename}:")
                for resultat in resultats:
                    print(f"  Texte similaire: {resultat['text'][:100]}..., Similarité: {resultat['similarity']}")
            else:
                print(f"Aucun plagiat détecté dans le fichier {filename}")

# Sauvegarder l'index et les embeddings (optionnel)
faiss.write_index(index, "corpus_index.bin")
with open('corpus_embeddings.pkl', 'wb') as f:
    pickle.dump(corpus_embeddings, f)

