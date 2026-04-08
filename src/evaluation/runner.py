import time
import pandas as pd
from typing import List, Dict
from src.evaluation.metrics import calc_exact_match, calc_rouge_bleu, PPLCalculator
from src.evaluation.judge import llm_as_a_judge

class EvaluationRunner:
    def __init__(self, engine):
        self.engine = engine
        self.ppl_calc = PPLCalculator()

    def run_full_evaluation(self, dataset: List[Dict]) -> pd.DataFrame:
        results = []
        for idx, data in enumerate(dataset):
            q, gt_num, gt_text = data["query"], data["gt_number"], data["gt_text"]
            print(f"\n▶️ [평가 {idx+1}/{len(dataset)}] {q}")
            
            methods = {
                "Method 0 (Baseline)": self.engine.run_method0_baseline,
                "Method 1 (Vision Real)": self.engine.run_method1_vision,
                "Method 2 (Dual Basic)": self.engine.run_method2_dual_basic,
                "Method 3 (No-Filter)": lambda q: self.engine.run_method3_sota(q, use_prefilter=False),
                "Method 3 (No-Window)": lambda q: self.engine.run_method3_sota(q, max_expansions=0),
                "Method 3 (SOTA)": self.engine.run_method3_sota
            }
            
            for model_name, method_func in methods.items():
                print(f"   🚀 [{model_name}] 시작...")
                pred_text = "ERROR: API Failure after retries"
                latency = 0.0
                max_attempts = 3
                
                for attempt in range(max_attempts):
                    try:
                        start_time = time.time()
                        pred_text = method_func(q)
                        latency = round(time.time() - start_time, 2)
                        break
                    except Exception as e:
                        wait = (attempt + 1) * 30  # 지수적 대기 시간 (30초, 60초, 90초)
                        print(f"   ⚠️ 오류 발생 ({model_name}): {e}. {wait}초 후 다시 시도... ({attempt + 1}/{max_attempts})")
                        time.sleep(wait)

                # 메트릭 계산 부문은 실패 문구("실패", "Error") 등을 고려하여 안전하게 처리
                rouge, bleu = calc_rouge_bleu(pred_text, gt_text)
                results.append({
                    "Query_ID": idx + 1,
                    "Type": data["type"],
                    "Model": model_name,
                    "Answer": pred_text,
                    "Exact_Match": calc_exact_match(pred_text, gt_num, data.get("unit")),
                    "ROUGE-L": round(rouge, 3),
                    "BLEU": round(bleu, 3),
                    "LLM_Judge": llm_as_a_judge(q, pred_text, gt_text),
                    "Latency (sec)": latency
                })
                time.sleep(2) # Rate Limit protection
            time.sleep(15) # Heavy sleep between queries for safety
        
        return pd.DataFrame(results)
