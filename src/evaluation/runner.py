import time
import pandas as pd
from typing import List, Dict
from src.evaluation.metrics import calc_exact_match, calc_rouge_bleu, PPLCalculator
from src.evaluation.judge import llm_as_a_judge

class EvaluationRunner:
    """각 평가 데이터셋을 순회하며 Method 0-3의 성능을 측정하는 메인 평가 클래스"""
    def __init__(self, engine):
        self.engine = engine
        self.ppl_calc = PPLCalculator()

    def run_full_evaluation(self, dataset: List[Dict]) -> pd.DataFrame:
        """데이터셋 전체에 대해 모든 RAG 방법론(Method 0-3)을 평가합니다."""
        return self.run_selective_evaluation(dataset, target_methods=None)

    def run_selective_evaluation(self, dataset: List[Dict], target_methods: List[str] = None) -> pd.DataFrame:
        """데이터셋에 대해 특정 RAG 방법론들만 선택적으로 평가합니다."""
        results = []
        for idx, data in enumerate(dataset):
            q, gt_num, gt_text = data["query"], data["gt_number"], data["gt_text"]
            print(f"\n[평가 {idx+1}/{len(dataset)}] {q}")
            
            all_methods = {
                "Method 0 (Baseline)": lambda q: self.engine.run_method0_baseline(q, return_metadata=True),
                "Method 1 (Vision Only)": lambda q: self.engine.run_method1_vision(q, return_metadata=True),
                "Method 2 (Dual Basic)": lambda q: self.engine.run_method2_dual_basic(q, return_metadata=True),
                "w/o Filter": lambda q: self.engine.run_method3_sota(q, use_prefilter=False, return_metadata=True),
                "w/o Reranker": lambda q: self.engine.run_method3_sota(q, use_reranker=False, return_metadata=True),
                "w/o Agent Window": lambda q: self.engine.run_method3_sota(q, max_expansions=0, return_metadata=True),
                "SOTA (Full)": lambda q: self.engine.run_method3_sota(q, return_metadata=True)
            }
            
            # 선택된 메서드만 필터링 (None이면 전체 실행)
            methods = {k: v for k, v in all_methods.items() if target_methods is None or k in target_methods}
            
            for model_name, method_func in methods.items():
                print(f"   [{model_name}] 시작...")
                pred_text = "ERROR: API Failure after retries"
                metadata = {}
                latency = 0.0
                max_attempts = 3
                
                for attempt in range(max_attempts):
                    try:
                        start_time = time.time()
                        res_data = method_func(q)
                        pred_text = res_data["answer"]
                        metadata = res_data["metadata"]
                        latency = round(time.time() - start_time, 2)
                        break
                    except Exception as e:
                        wait = (attempt + 1) * 30
                        print(f"   [!] 오류 발생 ({model_name}): {e}. {wait}초 후 다시 시도... ({attempt + 1}/{max_attempts})")
                        time.sleep(wait)

                rouge, bleu = calc_rouge_bleu(pred_text, gt_text)
                usage = metadata.get("usage", {})
                
                results.append({
                    "Query_ID": idx + 1,
                    "Type": data["type"],
                    "Model": model_name,
                    "Answer": pred_text,
                    "Exact_Match": calc_exact_match(pred_text, gt_num, data.get("unit")),
                    "ROUGE-L": round(rouge, 3),
                    "BLEU": round(bleu, 3),
                    "PPL": round(self.ppl_calc.calculate(pred_text), 2),
                    "LLM_Judge": llm_as_a_judge(q, pred_text, gt_text),
                    "Total_Tokens": usage.get("total_tokens", 0),
                    "Latency (sec)": latency
                })
                time.sleep(2)
            time.sleep(10)
        
        return pd.DataFrame(results)
