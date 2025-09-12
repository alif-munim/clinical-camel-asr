import re
from langsmith.schemas import Run, Example

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Using a standard algorithm for Levenshtein distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1,        # Deletion
                          d[i][j-1] + 1,        # Insertion
                          d[i-1][j-1] + cost)   # Substitution

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def think_tags_evaluator(run: Run, example: Example) -> dict:  
    """Evaluator that checks if the output contains <think> tags."""  
    try:  
        hypothesis = str(run.outputs.get('content', ''))
          
        # Make the check case-insensitive to match <think> or <THINK>
        has_think_tags = bool(re.search(r'<think>.*?</think>', hypothesis, flags=re.DOTALL | re.IGNORECASE))
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
    """
    Measures Word Error Rate on the content inside <answer> tags.
    Lower WER is better.
    """
    try:  
        hypothesis = str(run.outputs.get('content', ''))
        reference = str(example.outputs.get("ground_truth", ""))

        # --- NEW LOGIC ---
        # 1. Extract the content from inside the <answer> tags.
        answer_match = re.search(r'<answer>(.*?)</answer>', hypothesis, flags=re.DOTALL | re.IGNORECASE)
        
        extracted_answer = ""
        if answer_match:
            # If tags are found, use the content within them as the hypothesis.
            extracted_answer = answer_match.group(1).strip()
        else:
            # If <answer> tags are not found, the model failed the instruction.
            # We treat the hypothesis as empty, which will result in a 100% error rate.
            pass

        # 2. Calculate WER on the extracted answer.
        wer_score = calculate_wer(reference, extracted_answer)
        
        # 3. The final score is inverted: 1.0 for perfect (WER=0), 0.0 for total mismatch (WER=1).
        score = 1.0 - wer_score  
  
        return {  
            "key": "word_error_rate",  
            "score": score,  
            "comment": f"WER: {wer_score:.3f}. Found <answer> tags: {'Yes' if answer_match else 'No'}"
        }  
    except Exception as e:  
        return {  
            "key": "word_error_rate",  
            "score": 0.0,  
            "comment": f"Error in evaluation: {str(e)}"  
        }  
  
evaluators = [wer_evaluator, think_tags_evaluator]