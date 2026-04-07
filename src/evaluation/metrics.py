import nltk
import numpy as np
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer

def calc_exact_match(pred: str, gt_num: str) -> int:
    clean_pred = pred.replace(",", "").replace(" ", "")
    clean_gt = gt_num.replace(",", "").replace(" ", "")
    return 1 if clean_gt in clean_pred else 0

def calc_rouge_bleu(pred: str, gt_text: str):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = scorer.score(gt_text, pred)['rougeL'].fmeasure
    reference = [nltk.word_tokenize(gt_text)]
    candidate = nltk.word_tokenize(pred)
    bleu = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
    return rouge, bleu

class PPLCalculator:
    def __init__(self, model_id: str = "skt/kogpt2-base-v2"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
        except:
            self.tokenizer = self.model = None

    def calculate(self, text: str) -> float:
        if not self.model or not text.strip() or "실패" in text: return np.nan
        encodings = self.tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        with torch.no_grad():
            outputs = self.model(encodings.input_ids, labels=encodings.input_ids)
            return torch.exp(outputs.loss).item()
