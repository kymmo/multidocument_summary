import bm25s
from Stemmer import Stemmer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import logging

logging.getLogger('bm25s').setLevel(logging.WARNING)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InforMetricsCalculator:
     def __init__(self, TOP_K = 5, BM25_SCORE_MIN = 0.2, ENTAIL_THRESHOLD = 0):
          self.stemmer = Stemmer("english")
          self.nlp = spacy.load("en_core_web_lg")
          self.sw = self.nlp.Defaults.stop_words
          self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
          self.model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").eval()
          self.model.to(device)
          
          self.TOP_K = TOP_K
          self.BM25_SCORE_MIN = BM25_SCORE_MIN
          self.ENTAIL_THRESHOLD = ENTAIL_THRESHOLD

     def _get_infor_metrics(self, doc_list, gen_summary_list, ref_summary_list, BATCH_SIZE = 3):
          """
          Args:
               doc_list (_type_): list of doc sentences list
               gen_summary_list (_type_): string list. one string element for one summary
               ref_summary_list (_type_): string list. one string element for one summary
               BATCH_SIZE (int, optional): nlp process batch size.

          """
          if len(gen_summary_list) != len(ref_summary_list):
               raise ValueError("[ERROR] [InforMetrics] Generated Summary and Reference Summary have difference size!")
          
          hallucination_rates = []
          faithfulness_scores = []
          omission_rates = []
          strong_hallucinations = []
          contradictions = []
          
          gen_summary_docs = list(self.nlp.pipe(gen_summary_list, batch_size=BATCH_SIZE))
          ref_summary_docs = list(self.nlp.pipe(ref_summary_list, batch_size=BATCH_SIZE))
          gen_summary_sents_list = [[sent.text for sent in doc.sents] for doc in gen_summary_docs]
          ref_summary_sents_list = [[sent.text for sent in doc.sents] for doc in ref_summary_docs]
          
          for doc_sents, gen_summary_sents, ref_summary_sents in zip(doc_list, gen_summary_sents_list, ref_summary_sents_list):
               scores = self._get_doc_metrics(doc_sents, gen_summary_sents, ref_summary_sents)
               hallucination_rates.append(scores['hallucination'])
               faithfulness_scores.append(scores['faithfulness'])
               omission_rates.append(scores['omission'])
               strong_hallucinations.append(scores['strong_hallucination'])
               contradictions.append(scores['contradiction'])
          
          return {
               "hallucination": round(sum(hallucination_rates) / len(hallucination_rates), 4),
               "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
               "omission": round(sum(omission_rates) / len(omission_rates), 4),
               "strong_hallucination": round(sum(strong_hallucinations) / len(strong_hallucinations), 4),
               "contradiction": round(sum(contradictions) / len(contradictions), 4),
          }
     
     def _get_doc_metrics(self, doc_sent_list, gen_summary_sents, ref_summary_sents):
          """parameters should be plain sentence string list. no nest list.
          """
          corpus_tokens = bm25s.tokenize(
               doc_sent_list,
               stopwords=self.sw,
               stemmer=self.stemmer
          )

          retriever = bm25s.BM25(corpus=doc_sent_list)
          retriever.index(corpus_tokens)
          
          gen_ent, gen_contra = self._retrieve_and_nli(retriever, gen_summary_sents)
          hallucination, strong_hallucination = self._calculate_hallucination(gen_ent)
          omission = self._calculate_omission(gen_summary_sents, ref_summary_sents)
          faithfulness = self._calculate_faithfulness(gen_ent)
          
          return {
               "hallucination": round(hallucination, 4),
               "strong_hallucination": round(strong_hallucination, 4),
               "faithfulness": round(faithfulness, 4),
               "omission": round(omission, 4),
               "contradiction": round(sum(gen_contra)/len(gen_contra), 4) if gen_contra else 0.0
          }
          
     def _retrieve_and_nli(self, retriever, hypotheses):
          """For each hypothesis sentence, retrieve premises and compute max entailment prob."""
          q_tokens = bm25s.tokenize(
               hypotheses,
               stopwords=self.sw,
               stemmer=self.stemmer
          )
          results, scores = retriever.retrieve(q_tokens, k=min(self.TOP_K, len(q_tokens)))
          
          max_entail_probs = []
          max_contradiction_probs = []
          for i, hypo in enumerate(hypotheses):
               premise_list = []
               for premise, score in zip(results[i], scores[i]):
                    if score >= self.BM25_SCORE_MIN:
                         premise_list.append(premise)
               
               if not premise_list:
                    print("[Warning] [InforMetrics] premise_list is empty. ")
                    max_entail_probs.append(0.0)
                    max_contradiction_probs.append(0.0)
                    continue
               
               if len(premise_list) > 1:
                    combined_premise = " ".join(premise_list)
                    enc = self.tokenizer(
                         combined_premise,
                         hypo,
                         return_tensors="pt",
                         truncation=True,
                         padding=True,
                         max_length=512,
                    ).to(device)
               else:
                    enc = self.tokenizer(
                         premise_list,
                         [hypo] * len(premise_list),
                         return_tensors="pt",
                         truncation=True,
                         padding=True,
                         max_length=512,
                    ).to(device)
               
               with torch.no_grad():
                    logits = self.model(**enc).logits
                    probs = torch.softmax(logits, dim=-1)
                    
               max_entail_probs.append(probs[:, 2].max().item())  # entailment
               max_contradiction_probs.append(probs[:, 0].max().item())  # contradiction
          
          return max_entail_probs, max_contradiction_probs

     def _calculate_faithfulness(self, gen_ent):
          if not gen_ent:
               return 0.0
          
          base_faithfulness = sum(p >= self.ENTAIL_THRESHOLD for p in gen_ent) / len(gen_ent)
          
          return base_faithfulness
     
     def _calculate_omission(self, gen_summary_sents, ref_summary_sents):
          if not ref_summary_sents:
               return 1.0
               
          omission_scores = []
          gen_full_text = " ".join(gen_summary_sents)
          
          for ref_sent in ref_summary_sents:
               if ref_sent in gen_full_text:
                    omission_scores.append(0.0)
                    continue
                    
               inputs = self.tokenizer(
                    ref_sent,
                    gen_full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
               ).to(device)
               
               with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0]
                    entail_prob = probs[2].item()
               
               is_omitted = 1.0 if entail_prob < self.ENTAIL_THRESHOLD else 0.0
               omission_scores.append(is_omitted)
          
          return sum(omission_scores) / len(omission_scores)
     
     def _calculate_hallucination(self, gen_ent):
          if not gen_ent:
               return 0.0, 0.0
          
          strong_hallucination = sum(p < 0.3 for p in gen_ent) / len(gen_ent)
          weak_hallucination = sum(0.3 <= p < 0.6 for p in gen_ent) / len(gen_ent)
          
          hallucination_score = strong_hallucination * 1.0 + weak_hallucination * 0.5
          
          return hallucination_score, strong_hallucination