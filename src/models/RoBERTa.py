import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
import spacy


class SingleDocSummarizerBERT:

     def __init__(self, document, num_sentences=6, num_clusters=3):
          """Define the basic args of the class

          Args:
               document (string): The document to be summarized
               num_sentences (int, optional): Number of top sentences to extract. Defaults to 6.
               num_clusters (int, optional): Number of clusters for k-means. Defaults to 3.
          """
          self.document = document
          self.num_sentences = num_sentences
          self.num_clusters = num_clusters
          
     
     def sentences_score_cal(self, similarities_weight = 0.5, distance2centroid_weight = 0.5):
          """Calculate the importance score of each sentence using the weighted sum of the similary and the distance to centroid.
          similarity: the similarity between sentence embedding and the document embedding
          distance2centroid: the distance between the sentence embedding and the cluster center embedding. We use K means clustering.

          Args:
               similarities_weight (float, optional): the weight of the similarity. Defaults to 0.5.
               distance2centroid_weight (float, optional): the weight of the distence to the cluster center. Defaults to 0.5.

          Returns:
               sentences: the splitted sentence
               combined_scores: Return the score of corresponding sentence in the document
          """
          # split text into sentences
          sentence_model = SentenceTransformer('all-mpnet-base-v2')
          nlp = spacy.load("en_core_web_md")
          doc = nlp(self.document)
          sentences = [sent.text.strip() for sent in doc.sents]
          
          ### TODO: num_sentences and num_clusters need to be dynamic
          ### TODO: to score sentences with a model, like BERT
          
          if not sentences:
               return []

          if len(sentences) < self.num_sentences:
               return sentences
          
          sentence_embeddings = sentence_model.encode(sentences)
          document_embedding = np.mean(sentence_embeddings, axis=0)
          similarities = np.dot(sentence_embeddings, document_embedding) / (np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(document_embedding))

          # clustering sentences using KMeans
          kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init = 'auto')
          kmeans.fit(sentence_embeddings)
          clusters = kmeans.predict(sentence_embeddings)
          centroids = kmeans.cluster_centers_

          # get the distance from centroid of each sentence
          distance2centroid = []
          for i, embedding in enumerate(sentence_embeddings):
               centroid_for_sentence = centroids[clusters[i]]
               distance = np.linalg.norm(embedding - centroid_for_sentence)
               distance2centroid.append(distance)

          # distance normalization. range from 0-1
          distance2centroid = np.array(distance2centroid)
          if np.max(distance2centroid) - np.min(distance2centroid) != 0:
               distance2centroid = (distance2centroid - np.min(distance2centroid)) / (np.max(distance2centroid) - np.min(distance2centroid))
          else:
               distance2centroid = np.zeros(len(sentences))

          # similarity normalization. range from 0-1
          if np.max(similarities) - np.min(similarities) != 0:
               similarities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities))
          else:
               similarities = np.zeros(len(sentences))

          # weighted sum
          combined_scores = (similarities_weight * similarities) + (distance2centroid_weight * (1 - distance2centroid))
          
          
          return sentences, combined_scores


     def get_top_k_sentences(self):
          sentences, combined_scores = self.sentences_score_cal()
          top_sentence_indices = np.argsort(combined_scores)[::-1][:self.num_sentences]
          top_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
          return top_sentences
     
     def evaluate(self):
          #### TODO: to evaluate the model using SDS dataset.
          #### may be need smaller granularity to make sensible output
          return