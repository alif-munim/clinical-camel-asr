import re
from langsmith.schemas import Run, Example

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculates the Word Error Rate (WER) between a reference and a hypothesis string.
    The WER is a measure of transcription accuracy.
    """
    # Normalize strings: lowercase and split into words
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Handle edge case of empty reference string
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Initialize the distance matrix for dynamic programming (Levenshtein distance)
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    # Compute the Levenshtein distance
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1,        # Deletion
                          d[i][j-1] + 1,        # Insertion
                          d[i-1][j-1] + cost)   # Substitution

    # The final WER is the total number of errors divided by the number of words in the reference
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    
    # The error rate can be > 1.0 if there are many more insertions than reference words,
    # but for scoring purposes, we can cap it at 1.0.
    return min(wer, 1.0)


def wer_evaluator(run: Run, example: Example) -> dict:  
    """
    Measures Word Error Rate (WER) by comparing the model's entire output
    to the ground truth reference. A lower WER is better.
    """
    try:  
        # The entire model output is now considered the hypothesis.
        hypothesis = str(run.outputs.get('content', '')).strip()
        reference = str(example.outputs.get("ground_truth", "")).strip()
        
        wer_score = calculate_wer(reference, hypothesis)  
        
        # The final score is inverted for the optimizer:
        # A perfect transcript (WER=0) gets a score of 1.0.
        # A completely wrong transcript (WER=1.0) gets a score of 0.0.
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
            "comment": f"Error in wer_evaluator: {str(e)}"  
        }  
  
# The list of evaluators now only contains the simplified WER evaluator.
evaluators = [wer_evaluator]
