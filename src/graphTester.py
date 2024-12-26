import torch
import dgl

import os
from utils.calculator import eval_label
from utils.logger import *
from rouge import Rouge

class TestPipLine():
     def __init__(self, model, m, test_dir, limited):
          """
               :param model: the model
               :param m: the number of sentence to select
               :param test_dir: for saving decode files
               :param limited: for limited Recall evaluation
          """
          self.model = model
          self.limited = limited
          self.m = m
          self.test_dir = test_dir
          self.extracts = []

          self.batch_number = 0
          self.running_loss = 0
          self.example_num = 0
          self.total_sentence_num = 0

          self._hyps = []
          self._refer = []

     def evaluation(self, G, index, valset):
          pass

     def getMetric(self):
          pass

     def SaveDecodeFile(self):
          import datetime
          nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # 现在
          log_dir = os.path.join(self.test_dir, nowTime)
          with open(log_dir, "wb") as resfile:
               for i in range(self.rougePairNum):
                    resfile.write(b"[Reference]\t")
                    resfile.write(self._refer[i].encode('utf-8'))
                    resfile.write(b"\n")
                    resfile.write(b"[Hypothesis]\t")
                    resfile.write(self._hyps[i].encode('utf-8'))
                    resfile.write(b"\n")
                    resfile.write(b"\n")
                    resfile.write(b"\n")

     @property
     def running_avg_loss(self):
          return self.running_loss / self.batch_number

     @property
     def rougePairNum(self):
          return len(self._hyps)

     @property
     def hyps(self):
          if self.limited:
               hlist = []
               for i in range(self.rougePairNum):
                    k = len(self._refer[i].split(" "))
                    lh = " ".join(self._hyps[i].split(" ")[:k])
                    hlist.append(lh)
               return hlist
          else:
               return self._hyps

     @property
     def refer(self):
          return self._refer

     @property
     def extractLabel(self):
          return self.extracts

class T5Tester(TestPipLine):
     def __init__(self, model, m, test_dir=None, limited=False):
          super().__init__(model, m, test_dir, limited)
          self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

     def evaluation(self, G, index, dataset):
          """
               :param G: the model
               :param index: list, example id
               :param dataset: dataset which includes text and summary
          """
          self.batch_number += 1
          outputs = self.model.forward(G)
          # logger.debug(outputs)
          snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
          label = G.ndata["label"][snode_id].sum(-1)            # [n_nodes]
          G.nodes[snode_id].data["loss"] = self.criterion(outputs, label).unsqueeze(-1)    # [n_nodes, 1]
          loss = dgl.sum_nodes(G, "loss")    # [batch_size, 1]
          loss = loss.mean()
          self.running_loss += float(loss.data)

          G.nodes[snode_id].data["p"] = outputs
          glist = dgl.unbatch(G)
          for j in range(len(glist)):
               idx = index[j]
               example = dataset.get_example(idx)
               original_article_sents = example.original_article_sents
               sent_max_number = len(original_article_sents)
               refer = example.original_abstract

               g = glist[j]
               snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
               N = len(snode_id)
               p_sent = g.ndata["p"][snode_id]
               p_sent = p_sent.view(-1, 2)   # [node, 2]
               label = g.ndata["label"][snode_id].sum(-1).squeeze().cpu()    # [n_node]
               if self.m == 0:
                    prediction = p_sent.max(1)[1] # [node]
                    pred_idx = torch.arange(N)[prediction!=0].long()
               else:
                    topk, pred_idx = torch.topk(p_sent[:,1], min(self.m, N))

                    prediction = torch.zeros(N).long()
                    prediction[pred_idx] = 1
               self.extracts.append(pred_idx.tolist())

               hyps = "\n".join(original_article_sents[id] for id in pred_idx if id < sent_max_number)

               self._hyps.append(hyps)
               self._refer.append(refer)

     def getMetric(self):
          logger.info("[INFO] Start calculating rouge scores...")
          rouge = Rouge()
          scores_all = rouge.get_scores(self.hyps, self.refer, avg=True)
          res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
               scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
                    + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
               scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
                    + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
               scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
          logger.info(res)