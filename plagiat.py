import os
from transformers import BertTokenizer, BertModel
import torch
import faiss
import numpy as np
from sentence_transformers import util
import chardet 
from logging.handlers import RotatingFileHandler
import fitz
from io import BytesIO

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


class Plagiat:
    def detect_encoding(self, file_stream):
        print("=========detect_encoding START=========")
        file_stream.seek(0)  # Réinitialiser le curseur après lecture
        raw_data = file_stream.read()
       
        result = chardet.detect(raw_data)
        return result['encoding']

    def embed_bert(self,text, tokenizer, model):
        print("=========embed_bert START=========")
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[0, 1:-1, :].mean(dim=0).numpy()
        return embeddings

    def check_plagiarism_detailed(self,text, tokenizer, model, index, corpus, corpus_embeddings, threshold=0.8):
        print("=========check_plagiarism_detailed START=========")
        plagiarized_parts = []
        sentences = text.split(". ")  # Split the text into sentences for comparison
        for sentence in sentences:
            sentence_embedding = self.embed_bert(sentence, tokenizer, model)
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

    def save_output_data(self,corpus_embeddings, corpus, dossier_copie):
        print("=========save_output_data START=========")
        index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
        index.add(corpus_embeddings)

        # Create a file to save results
        with open("resultats_plagiat.txt", "w", encoding='utf-8') as result_file:
            for filename in os.listdir(dossier_copie):
                if filename.endswith(".txt"):
                    file_path = os.path.join(dossier_copie, filename)
                    with open(file_path, 'rb') as file_stream:
                        encoding = self.detect_encoding(file_stream)
                    with open(file_path, 'r', encoding=encoding) as f:
                        texte_copie = f.read()
                        resultats = self.check_plagiarism_detailed(
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
                elif filename.endswith(".pdf"):
                    file_path = os.path.join(dossier_copie, filename)
                    texte_copie = self.extract_text_from_pdf(file_path)
                    resultats = self.check_plagiarism_detailed(
                        texte_copie, tokenizer, model, index, corpus, corpus_embeddings
                    )

                    part_plagiat = ''
                    match_text = ''
                    i = 0
                    similarity = 0
                    if resultats:
                        result_file.write(f"Plagiat détecté dans le fichier {filename}:\n")
                        for resultat in resultats:
                            part_plagiat += f"  Partie plagiée: {resultat['sentence']}\n"
                            match_text += f"  Correspondance trouvée: {resultat['matched_text'][:100]}...\n"
                            similarity += resultat['similarity']
                            i += 1

                        
                            # print(f"  Partie plagiée: {resultat['sentence']}")
                            # print(f"  Correspondance trouvée: {resultat['matched_text'][:100]}...")
                            # print(f"  Similarité: {resultat['similarity']:.2f}")
                        result_file.write(f"  Partie plagiée: {resultat['sentence']}\n")
                        result_file.write(f"  Correspondance trouvée: {resultat['matched_text'][:100]}...\n")
                        result_file.write(f"  Similarité: {similarity / i}\n\n")
                    else:
                        result_file.write(f"Aucun plagiat détecté dans le fichier {filename}.\n")
                        # print(f"Aucun plagiat détecté dans le fichier {filename}")
        print("finish")

    def extract_text_from_pdf_byte(self,file_byte):
        print("=========extract_text_from_pdf START=========")
        doc = fitz.open(stream=file_byte, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def extract_text_from_pdf(self,pdf_file):
        print("=========extract_text_from_pdf START=========")
        doc = fitz.open(pdf_file)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def detect_plagiat(self,file, file_type):
        print("=========detect_plagiat START=========")

            
        corpus = []
        corpus_embeddings = []
        file_byte = BytesIO(file.body)
        if file_type == 'txt':
            dossier_copie = "TestData1/Copy/"
            
            file_byte = BytesIO(file.body)  # Assurez-vous que `file` est bien un objet avec `.body` 
            encoding = self.detect_encoding(file_byte)
            file_byte.seek(0)
            raw_data = file_byte.read()
            texte_original = raw_data.decode(encoding)
            corpus.append(texte_original)
            corpus_embeddings.append(self.embed_bert(texte_original, tokenizer, model))

        elif type == 'pdf':
            dossier_copie = "testPDF"
            texte_original = self.extract_text_from_pdf_byte(file_byte)
            corpus.append(texte_original)
            corpus_embeddings.append(self.embed_bert(texte_original, tokenizer, model))

        corpus_embeddings = np.array(corpus_embeddings)

        self.save_output_data(corpus_embeddings, corpus, dossier_copie)
