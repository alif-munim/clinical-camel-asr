from langsmith.schemas import Run, Example
import re

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def think_tags_evaluator(run: Run, example: Example) -> dict:  
    """Evaluator that checks if the output contains <think> tags."""  
    try:  
        output_dict = run.outputs  
        hypothesis = str(output_dict['content'])  
          
        has_think_tags = '<think>' in hypothesis and '</think>' in hypothesis  
        score = 1.0 if has_think_tags else 0.0  
          
        return {  
            "key": "think_tags_present",  
            "score": score,  
            "comment": "Pass: Contains <think> tags" if has_think_tags else "Fail: Missing <think> tags"  
        }  
    except Exception as e:  
        return {  
            "key": "think_tags_present",  
            "score": 0.0,  
            "comment": f"Error in evaluation: {str(e)}"  
        }

def wer_evaluator(run: Run, example: Example) -> dict:  
    """Evaluator that measures Word Error Rate. Lower WER is better."""  
    try:  
        output_dict = run.outputs  
        hypothesis = str(output_dict['content'])  
        reference = str(example.outputs.get("ground_truth", ""))

        # --- START DEBUG PRINTS ---
        # print("───────────────────────────────────")
        # print("HYPOTHESIS (Model Output):")
        # print(repr(hypothesis))
        # print("\nREFERENCE (Ground Truth):")
        # print(repr(reference))
        # print("───────────────────────────────────")
        # --- END DEBUG PRINTS ---  
          
        # Strip <think> tags and their content from hypothesis  
        # hypothesis_clean = re.sub(r'<think>.*?</think>', '', hypothesis, flags=re.DOTALL).strip()  
        hypothesis_clean = re.sub(r'<think>.*?</think>', '', hypothesis, flags=re.DOTALL | re.IGNORECASE).strip()
          
        wer_score = calculate_wer(reference, hypothesis_clean)  
        score = 1.0 - wer_score  
  
        return {  
            "key": "word_error_rate",  
            "score": score,  
            "comment": f"WER: {wer_score:.3f}"  
        }  
    except Exception as e:  
        return {  
            "key": "word_error_rate",  
            "score": 0.0,  
            "comment": f"Error in evaluation: {str(e)}"  
        }  
  
evaluators = [wer_evaluator, think_tags_evaluator]