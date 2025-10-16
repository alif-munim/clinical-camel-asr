# test_soap_note.py  
import os
import pytest  
import json  
from deepeval import assert_test  
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  
from deepeval.metrics import GEval  
  
# Configuration - modify these paths as needed  
TRANSCRIPT_PATH = os.getenv("TRANSCRIPT_PATH", "transcripts/patient1_transcript.txt")  
NOTE_PATH = os.getenv("NOTE_PATH", "notes/patient1_note.json")
  
def parse_json_soap_note(note_content):  
    """Parse JSON SOAP note into sections"""  
    try:  
        note_data = json.loads(note_content)  
    except json.JSONDecodeError as e:  
        raise ValueError(f"Invalid JSON in note file: {e}")  
      
    sections = {}  
    key_mapping = {  
        "Chief Complaint": "CC",  
        "History of Present Illness": "HPI",  
        "Impression": "Impression",  
        "Plan": "Plan"  
    }  
      
    for json_key, section_name in key_mapping.items():  
        if json_key in note_data:  
            if isinstance(note_data[json_key], list):  
                sections[section_name] = " ".join(note_data[json_key])  
            else:  
                sections[section_name] = note_data[json_key]  
      
    return sections  
  
def create_section_metric(section_name):  
    """Create a faithfulness metric for each SOAP section"""  
    return GEval(  
        name=f"{section_name} Faithfulness",  
        evaluation_steps=[  
            f"Extract all medical claims and information from the {section_name} section",  
            "Verify each claim against the doctor-patient transcript",  
            "Identify any contradictions or unsupported claims",  
            "Heavily penalize hallucinations or information not present in the transcript",  
            "Award high scores for accurate, complete information that aligns with the transcript"  
        ],  
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],  
        threshold=0.7,  
        model="gpt-5"  
    )  
  
# Load data once at module level  
with open(TRANSCRIPT_PATH, 'r') as f:  
    transcript = f.read()  
  
with open(NOTE_PATH, 'r') as f:  
    note_content = f.read()  
  
sections = parse_json_soap_note(note_content)  
  
# Create test cases for parametrization  
test_data = []  
for section_name in ['CC', 'HPI', 'Impression', 'Plan']:  
    if section_name in sections:  
        test_case = LLMTestCase(  
            input=f"Generate {section_name} from transcript",  
            actual_output=sections[section_name],  
            retrieval_context=[transcript]  
        )  
        metric = create_section_metric(section_name)  
        test_data.append((section_name, test_case, metric))  
  
# Pytest parametrized test function  
@pytest.mark.parametrize("section_name,test_case,metric", test_data)  
def test_soap_section(section_name, test_case, metric):  
    """Test each SOAP section for faithfulness to transcript"""  
    print(f"\nEvaluating {section_name}")  
    assert_test(test_case, [metric])  
    score_0_to_100 = metric.score * 100  
    print(f"{section_name} Score: {score_0_to_100:.1f}/100")