# clinical-canary

Set up your environment and API keys.
```
conda activate clinical_camel
export HOME_DIR=/home/bowang/Documents/alif/clinical-camel-asr
export OPENROUTER_API_KEY=<api-key>
export OPENAI_API_KEY=<api-key>
```


### LLM-as-a-Judge Eval
```
cd llm_judge
TRANSCRIPT_PATH=transcripts/patient1_transcript.txt NOTE_PATH=notes/patient1_note.json deepeval test run test_soap_note.py 2>&1 | tee results/patient1_output.log
```


## Prompt Optimization

Create a new conda environment and set your `LANGSMITH_API_KEY`
```
export LANGSMITH_API_KEY=<your-key>
cd <your-home-dir>
conda create -n promptim python=3.12
cp /home/bowang/Documents/alif/promptimizer <your-home-dir>
cd <your-home-dir>/promptimizer
pip install -e .
```

Use existing environment
```
conda activate promptim_v2
```

Copy an existing prompt optimization task and create your own.
```
cp prompt_optimization/pp-v2 prompt_optimization/<new-task>
```

Modify `config.json`, specifically the `dataset` and `prompt_str` fields. Start with a simple `prompt_str`.

Set up your dataset on https://smith.langchain.com/. Then set it to public and get the link. It will look something like this: https://smith.langchain.com/public/6ed521df-c0d8-42b7-a0db-48dd73a0c680/d <br>

Take note of your input and output fields when uploading your CSV. These will need to be changed in `task.py`, as well as the evaluators (if you aren't optimizing for WER) <br>

Once everything is ready, start training.
```
promptim train --task ./<your-task>/config.json
```

## Data Generation
```
python data_generation/json_to_transcript.py
```

```
python data_generation/generate_prompts.py --csv-path data_generation/new8.csv
```

## Evals
```
python evals/evaluate_ner.py /home/bowang/Documents/alif/clinical-camel-asr/data/whisper_0725.csv /home/bowang/Documents/alif/clinical-camel-asr/data/uhn19_gliner_v2.csv --ner_model gliner

python evals/evaluate_ner.py /home/bowang/Documents/alif/clinical-camel-asr/data/whisper_0725.csv /home/bowang/Documents/alif/clinical-camel-asr/data/uhn19_ompharma.csv --ner_model openmed-pharma

python evals/evaluate_ner.py /home/bowang/Documents/alif/clinical-camel-asr/data/whisper_0725.csv /home/bowang/Documents/alif/clinical-camel-asr/data/uhn19_ompatho.csv --ner_model openmed-pathology

python evals/evaluate_ner.py /home/bowang/Documents/alif/clinical-camel-asr/data/whisper_0725.csv /home/bowang/Documents/alif/clinical-camel-asr/data/uhn19_omana.csv --ner_model openmed-anatomy

python evals/evaluate_ner.py /home/bowang/Documents/alif/clinical-camel-asr/data/whisper_0725.csv /home/bowang/Documents/alif/clinical-camel-asr/data/uhn19_omgen.csv --ner_model openmed-genomics

python evals/evaluate_ner.py /home/bowang/Documents/alif/clinical-camel-asr/data/whisper_0725.csv /home/bowang/Documents/alif/clinical-camel-asr/data/uhn19_omonc.csv --ner_model openmed-oncology
```


## Quickstart on Workstation

Transcribe all of your audio files (if you have m4a files, use the `m4a_to_wav.py` script).
```
python transcribe_turbo.py --input_dir $HOME_DIR/wav_audio --output_dir $HOME_DIR/0624_audio_transcripts_vad --use_vad
```

Enhanced VAD settings.
```
python transcribe_turbo.py \
  --input_dir $HOME_DIR/wav_audio \
  --output_dir $HOME_DIR/0630_audio_transcripts \
  --use_vad
```



Postprocess and summarize with the model of your choice.
```
export RUN_DATE=0529 # or whatever identifier you want to use
export RUN_MODEL=llama-3.1-8b # other options are deepseek, qwq-32b, and qwen-72b
python postprocess_and_summarize.py \
  --model        "$RUN_MODEL" \
  --input_dir    "${RUN_DATE}_audio_transcripts" \
  --post_dir     "${RUN_DATE}_postprocessed_${RUN_MODEL}" \
  --summary_dir  "${RUN_DATE}_summarized_${RUN_MODEL}" \
  --post_prompt prompts/postprompt_v1.txt \
  --summary_prompt prompts/summaryprompt_v1.txt \
  --max_tokens 10000
```

Evaluate all of your summaries against ground truth transcripts.
```
export RUN_DATE=0530 # or whatever identifier you want to use
export RUN_MODEL=phi-4 # other options are deepseek, qwq-32b, and qwen-72b
ROOT=/home/jma/Documents/clinical-canary
python eval_batch.py \
  --prompt       "$ROOT/prompts/eval_v1.txt" \
  --gt-dir       "$ROOT/${RUN_DATE}_audio_transcripts" \
  --sum-dir      "$ROOT/${RUN_DATE}_summarized_${RUN_MODEL}" \
  --results-dir  "$ROOT/${RUN_DATE}_eval_results_${RUN_MODEL}" \
  --prompts-dir  "$ROOT/${RUN_DATE}_eval_prompts_${RUN_MODEL}" \
  --api-base     "https://api.openai.com/v1" \
  --temperature  0
```

## HuggingFace

If you don't specify `--or_model`, the same HF model will be used for postprocessing and summarization.
```
export RUN_DATE=0504 # or whatever identifier you want to use
export RUN_MODEL=medgemma-4b 
python postprocess_and_summarize_hf.py \
  --or_model qwen3-32b \
  --hf_model "$RUN_MODEL" \
  --input_dir  "${RUN_DATE}_audio_transcripts" \
  --post_dir   "${RUN_DATE}_postprocessed_qwen32b" \
  --summary_dir "${RUN_DATE}_summarized_medgemma27b" \
  --post_prompt    prompts/postprompt_v1.txt \
  --summary_prompt prompts/summaryprompt_v1.txt \
  --max_tokens 8192 \
  --temperature 0 \
  --skip_post
```

Evaluate all of your summaries against ground truth transcripts.
```
export RUN_MODEL=medgemma27b 
ROOT=/home/jma/Documents/clinical-canary
python eval_batch.py \
  --prompt       "$ROOT/prompts/eval_v1.txt" \
  --gt-dir       "$ROOT/${RUN_DATE}_audio_transcripts" \
  --sum-dir      "$ROOT/${RUN_DATE}_summarized_${RUN_MODEL}" \
  --results-dir  "$ROOT/${RUN_DATE}_eval_results_${RUN_MODEL}" \
  --prompts-dir  "$ROOT/${RUN_DATE}_eval_prompts_${RUN_MODEL}" \
  --api-base     "https://api.openai.com/v1" \
  --temperature  0
```

## Misc

turbo with VAD
```
python transcribe_turbo.py --input_dir /home/jma/Documents/clinical-canary/0424_audio_wav --output_dir /home/jma/Documents/clinical-canary/0424_audio_transcripts --use_vad
```

postprocess + summarize
```
export OPENROUTER_API_KEY=<api-key>
export RUN_DATE=0504  
export RUN_MODEL=gemma-27b
python postprocess_and_summarize.py \
  --model        "$RUN_MODEL" \
  --input_dir    "${RUN_DATE}_audio_transcripts" \
  --post_dir     "${RUN_DATE}_postprocessed_${RUN_MODEL}" \
  --summary_dir  "${RUN_DATE}_summarized_${RUN_MODEL}" \
  --post_prompt prompts/postprompt_v1.txt \
  --summary_prompt prompts/summaryprompt_v1.txt \
  --max_tokens 64000
```

evaluate single example with openai
```
export OPENAI_API_KEY=<api-key>
python eval_batch.py \
  --prompt /home/jma/Documents/clinical-canary/prompts/eval_v1.txt \
  --single-mode \                                                        
  --transcript "/home/jma/Documents/clinical-canary/0424_audio_transcripts/15 - Radiation encounter - Radiation Therapist - Audio Claude.txt" \                                              
  --summary "/home/jma/Documents/clinical-canary/0424_summarized_deepseek/15 - Radiation encounter - Radiation Therapist - Audio Claude.sum.txt" \
  --results-dir /home/jma/Documents/clinical-canary/0424_eval_single_test \
  --prompts-dir /home/jma/Documents/clinical-canary/0424_eval_prompts \
  --api-base https://api.openai.com/v1 \
  --temperature 0
```

batch evaluate with openai (pass the gt and summary directories)
```
export OPENAI_API_KEY=<api-key>
export RUN_DATE=0504          
export RUN_MODEL=gemma-27b
ROOT=/home/jma/Documents/clinical-canary
python eval_batch.py \
  --prompt       "$ROOT/prompts/eval_v1.txt" \
  --gt-dir       "$ROOT/${RUN_DATE}_audio_transcripts" \
  --sum-dir      "$ROOT/${RUN_DATE}_summarized_${RUN_MODEL}" \
  --results-dir  "$ROOT/${RUN_DATE}_eval_results_${RUN_MODEL}" \
  --prompts-dir  "$ROOT/${RUN_DATE}_eval_prompts_${RUN_MODEL}" \
  --api-base     "https://api.openai.com/v1" \
  --temperature  0
```

# Primock 57

whisper
```
python whisper_transcribe.py --audio "/home/jma/Documents/vita/primock57/audio/day1_consultation01_doctor.wav" --textgrid "/home/jma/Documents/vita/primock57/transcripts/day1_consultation01_doctor.TextGrid" --model "base" --output "whisper_transcript.txt" --tier "Doctor"
```

whisper
```
python whisper_batch.py --audio_dir "/home/jma/Documents/vita/primock57/audio" \
                            --textgrid_dir "/home/jma/Documents/vita/primock57/transcripts" \
                            --output_dir "whisper_results" \
                            --model "base" \
                            --tier "Doctor"
```

distil-whisper
```
python distil_whisper_primock.py --audio_dir "/home/jma/Documents/vita/primock57/audio" \
                                     --textgrid_dir "/home/jma/Documents/vita/primock57/transcripts" \
                                     --model_type distil-large-v3.5 \
                                     --output_dir "distil_whisper_results"
```

whisper large v3 turbo (faster without chunk length / batch size)
```
python distil_whisper_primock.py --audio_dir "/home/jma/Documents/vita/primock57/audio" \
                                     --textgrid_dir "/home/jma/Documents/vita/primock57/transcripts" \
                                     --model_type whisper-large-v3-turbo \
                                     --output_dir "whisper-large-v3-turbo_results" \
                                     --chunk_length 30 --batch_size 8
```

whisper large
```
python distil_whisper_primock.py --audio_dir "/home/jma/Documents/vita/primock57/audio" \
                                     --textgrid_dir "/home/jma/Documents/vita/primock57/transcripts" \
                                     --model_type whisper-large-v3 \
                                     --output_dir "whisper-large-v3_results" \
                                     --chunk_length 30 --batch_size 8
```

audio lengths
```
python add_audio_lengths.py --csv "distil_whisper_results/wer_results.csv" \
                           --audio_dir "/home/jma/Documents/vita/primock57/audio" \
                           --analyze
```


# SciData 22

scidata distil
```
python distil_whisper_scidata.py --audio_dir "/home/jma/Documents/vita/scidata22-audio/Audio Recordings" --transcript_dir "/home/jma/Documents/vita/scidata22-audio/Clean Transcripts" --output_dir "results"
```

scidata whisper-large-v3-turbo
```
python distil_whisper_scidata.py --audio_dir "/home/jma/Documents/vita/scidata22-audio/Audio Recordings" --transcript_dir "/home/jma/Documents/vita/scidata22-audio/Clean Transcripts" --output_dir "results" --model_type whisper-large-v3-turbo
```

scidata whisper-large-v3
```
python distil_whisper_scidata.py --audio_dir "/home/jma/Documents/vita/scidata22-audio/Audio Recordings" --transcript_dir "/home/jma/Documents/vita/scidata22-audio/Clean Transcripts" --output_dir "results" --model_type whisper-large-v3
```

generate prompts and send to api
```
python generate_prompts.py --api-key="your-api-key-here" --max-retries=5 --retry-delay=10 --limit=2
python generate_prompts.py --max-retries=5 --retry-delay=10 --csv-path data/dialogue_list2.csv
python generate_prompts.py --no-api --csv-path data/dialogue_list2.csv --limit=2
```

# NVidia Canary

canary
```
python scripts/speech_to_text_aed_chunked_infer.py \
    pretrained_name="nvidia/canary-1b" \
    audio_dir="audio/" \
    output_filename="results/day01_doctor_output.json" \
    chunk_len_in_secs=40.0 \
    batch_size=1 \
    decoding.beam.beam_size=1
```