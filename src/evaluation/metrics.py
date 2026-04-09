import nltk
import numpy as np
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer

# 단위별 '원' 환산 배수 테이블
_UNIT_MULTIPLIER = {
    "원":     1,
    "천원":   1_000,
    "백만원": 1_000_000,
    "십억원": 1_000_000_000,
    "억원":   100_000_000,
    "조원":   1_000_000_000_000,
    "억":     100_000_000,
    "조":     1_000_000_000_000,
    "만":     10_000,
}

def _normalize_financial_number(s: str) -> str:
    """재무제표 숫자 표기를 정규화합니다.
    - 쉼표, 공백 제거
    - 괄호 음수 표기 (584,875) → -584875 변환
    """
    import re
    s = s.replace(",", "").replace(" ", "")
    # (숫자) 형태를 -숫자 형태로 변환: e.g. (584875) → -584875
    s = re.sub(r'\((\d+)\)', r'-\1', s)
    return s

def _to_base_value(value_str: str, unit: str | None) -> float | None:
    """숫자 문자열과 단위를 받아 '원' 기준의 float로 변환합니다."""
    import re
    clean = _normalize_financial_number(value_str)
    # '140만' 같이 단위가 섞인 경우 처리
    m_man = re.search(r'(\d+)만', clean)
    if m_man:
        val = float(m_man.group(1)) * 10_000
    else:
        m = re.search(r'-?[\d.]+', clean)
        if not m: return None
        val = float(m.group())
    
    multiplier = _UNIT_MULTIPLIER.get(unit, 1) if unit else 1
    return val * multiplier

def _extract_unit_and_value(text: str) -> tuple[float | None, str | None]:
    """모델 답변 텍스트에서 숫자와 단위를 추출합니다."""
    import re
    # [최종 정답]: 수치 패턴을 최우선으로 찾음
    m_final = re.search(r'\[최종 정답\]:\s*(-?[\d,.]+\s*[가-힣]+)', text)
    target_text = m_final.group(1) if m_final else text

    units = sorted(_UNIT_MULTIPLIER.keys(), key=len, reverse=True)
    pattern = r'(-?[\d,]+(?:\.\d+)?)\s*(' + '|'.join(units) + r')'
    m = re.search(pattern, target_text)
    if m:
        num_str = m.group(1).replace(',', '')
        unit = m.group(2)
        try:
            val = float(num_str)
            return val * _UNIT_MULTIPLIER[unit], unit
        except ValueError:
            return None, None
    return None, None

def calc_exact_match(pred: str, gt_num: str, unit: str | None = None) -> int:
    """모델 답변(pred)에 정답 숫자(gt_num)가 포함되어 있는지 확인합니다.

    평가 순서:
      1. 문자열 포함 여부 (기존 방식, 단위 무관)
      2. unit이 제공된 경우, 숫자를 '원' 단위로 환산하여 수치 비교
         → 모델이 다른 단위로 답해도 같은 금액이면 정답 처리

    재무제표 음수 표기 관행 처리:
      - 표준 음수: -584,875
      - 괄호 음수: (584,875)  ← 재무제표 원문 표기
    """
    # Step 1: 기존 문자열 기반 비교
    clean_pred = _normalize_financial_number(pred)
    clean_gt   = _normalize_financial_number(gt_num)
    if clean_gt in clean_pred:
        return 1

    # Step 2: 단위 기반 수치 비교 (unit이 있고, 숫자형 값인 경우)
    if unit and unit in _UNIT_MULTIPLIER:
        gt_base = _to_base_value(gt_num, unit)
        pred_base, _ = _extract_unit_and_value(pred)
        if gt_base is not None and pred_base is not None:
            if abs(gt_base - pred_base) < 1.0:  # 1원 미만 오차 허용
                return 1

    return 0

def calc_rouge_bleu(pred: str, gt_text: str):
    """결과 텍스트와 정답 텍스트 간의 형태적 유사도(ROUGE-L, BLEU)를 측정합니다."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = scorer.score(gt_text, pred)['rougeL'].fmeasure
    reference = [nltk.word_tokenize(gt_text)]
    candidate = nltk.word_tokenize(pred)
    bleu = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
    return rouge, bleu

class PPLCalculator:
    """문장 생성 품질을 측정하기 위해 Perplexity(PPL)를 계산하는 클래스"""
    def __init__(self, model_id: str = "skt/kogpt2-base-v2"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
        except:
            self.tokenizer = self.model = None

    def calculate(self, text: str) -> float:
        """입력 문장의 Perplexity 값을 산출합니다."""
        if not self.model or not text.strip() or "실패" in text: return np.nan
        encodings = self.tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        with torch.no_grad():
            outputs = self.model(encodings.input_ids, labels=encodings.input_ids)
            return torch.exp(outputs.loss).item()
