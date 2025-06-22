import bm25s
from Stemmer import Stemmer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import logging

logging.getLogger('bm25s').setLevel(logging.WARNING)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InforMetricsCalculator:
     def __init__(self, TOP_K=5, BM25_SCORE_MIN=0.1, ENTAIL_THRESHOLD=0.75, WEAK_HALLU_MIN = 0.25, WEAK_HALLU_MAX = 0.65):
          self.stemmer = Stemmer("english")
          self.nlp = spacy.load("en_core_web_lg")
          self.sw = self.nlp.Defaults.stop_words
          self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
          self.model = AutoModelForSequenceClassification.from_pretrained(
               "microsoft/deberta-v2-xlarge-mnli"
          ).eval()
          self.model.to(device)
          
          self.TOP_K = TOP_K
          self.BM25_SCORE_MIN = BM25_SCORE_MIN
          self.ENTAIL_THRESHOLD = ENTAIL_THRESHOLD
          self.WEAK_HALLU_MIN = WEAK_HALLU_MIN
          self.WEAK_HALLU_MAX = WEAK_HALLU_MAX

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
          cleaned_doc_sents = [
               sent for sent in doc_sent_list
               if sent.strip() and len(sent.split()) > 2
          ]
          corpus_tokens = bm25s.tokenize(
               cleaned_doc_sents,
               stopwords=self.sw,
               stemmer=self.stemmer
          )

          retriever = bm25s.BM25(corpus=cleaned_doc_sents)
          retriever.index(corpus_tokens)
          
          gen_ent, gen_contra = self._retrieve_and_nli(retriever, gen_summary_sents)
          hallucination, strong_hallucination = self._calculate_hallucination(gen_ent)
          omission = self._calculate_omission(gen_summary_sents, ref_summary_sents)
          faithfulness = self._calculate_faithfulness(gen_ent)
          contradiction_rate = sum(1 for p in gen_contra if p > 0.5) / len(gen_contra) if gen_contra else 0.0
          
          return {
               "hallucination": round(hallucination, 4),
               "strong_hallucination": round(strong_hallucination, 4),
               "faithfulness": round(faithfulness, 4),
               "omission": round(omission, 4),
               "contradiction": round(contradiction_rate, 4)
          }
          
     def _retrieve_and_nli(self, retriever, hypotheses):
          """For each hypothesis sentence, retrieve premises and compute max entailment prob."""
          q_tokens = bm25s.tokenize(
               hypotheses,
               stopwords=self.sw,
               stemmer=self.stemmer
          )
          results, scores = retriever.retrieve(q_tokens, k=min(self.TOP_K, len(retriever.corpus)))
          
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
               
               enc = self.tokenizer(
                    premise_list,
                    [hypo] * len(premise_list),
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_attention_mask=True,
               )
               enc = {k: v.to(device) for k, v in enc.items()}
               
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
     
     def _calculate_hallucination(self, gen_ent):
          if not gen_ent:
               return 0.0, 0.0
          
          strong_hallucination = sum(p < self.WEAK_HALLU_MIN for p in gen_ent) / len(gen_ent)
          weak_hallucination = sum(self.WEAK_HALLU_MIN <= p < self.WEAK_HALLU_MAX for p in gen_ent) / len(gen_ent)
          
          hallucination_score = strong_hallucination * 1.0 + weak_hallucination * 0.5
          
          return hallucination_score, strong_hallucination
     
     def _calculate_omission(self, gen_sents, ref_sents):
          if not ref_sents:
               return 0.0
          
          if not gen_sents:
               return 1.0
          
          gen_tokens = bm25s.tokenize(
               gen_sents,
               stopwords=self.sw,
               stemmer=self.stemmer
          )
          gen_retriever = bm25s.BM25(corpus=gen_sents)
          gen_retriever.index(gen_tokens)
          
          covered_count = 0
          ##sentence level compare
          for ref_sent in ref_sents:
               q_tokens = bm25s.tokenize(
                    [ref_sent],
                    stopwords=self.sw,
                    stemmer=self.stemmer
               )
               results, scores = gen_retriever.retrieve(q_tokens, k=min(self.TOP_K, len(gen_sents)))
               
               candidates = []
               for i, score in enumerate(scores[0]):
                    if score >= self.BM25_SCORE_MIN:
                         candidates.append(gen_sents[results[0][i]])
               
               if not candidates:
                    continue
               
               inputs = self.tokenizer(
                    candidates,
                    [ref_sent] * len(candidates),
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
               )
               inputs = {k: v.to(device) for k, v in inputs.items()}
               
               with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)
                    entail_probs = probs[:, 2]
               
               if (entail_probs >= self.ENTAIL_THRESHOLD).any():
                    covered_count += 1
          
          omission_rate = 1 - (covered_count / len(ref_sents))
          
          return round(omission_rate, 4)

     def _check_entity_consistency(self, gen_sents, doc_sents):
          doc_entities = set()
          for sent in doc_sents:
               doc = self.nlp(sent)
               doc_entities.update([(ent.text, ent.label_) for ent in doc.ents])
          
          inconsistent = 0
          total = 0
          for sent in gen_sents:
               doc = self.nlp(sent)
               for ent in doc.ents:
                    total += 1
                    if (ent.text, ent.label_) not in doc_entities:
                         if not self._is_synonym_entity(ent.text, ent.label_, doc_entities):
                              inconsistent += 1
          
          return inconsistent / total if total > 0 else 0.0

     def _is_synonym_entity(self, entity, label, doc_entities):
          entity_doc = self.nlp(entity)
          for doc_ent, doc_label in doc_entities:
               if label != doc_label:
                    continue
                    
               doc_ent_doc = self.nlp(doc_ent)
               if entity_doc.similarity(doc_ent_doc) > 0.85:
                    return True
          return False
