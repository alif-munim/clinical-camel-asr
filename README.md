# clinical-canary

turbo with VAD
```
python transcribe_turbo.py --input_dir /home/jma/Documents/clinical-canary/0424_audio_wav --output_dir /home/jma/Documents/clinical-canary/0424_audio_transcripts --use_vad
```

postprocess + summarize
```
export OPENROUTER_API_KEY=<api-key>
python postprocess_and_summarize.py \
  --model deepseek \
  --input_dir 0424_audio_transcripts \
  --post_dir 0424_postprocessed_deepseek \
  --summary_dir 0424_summarized_deepseek \
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
python eval_batch.py \
  --prompt /home/jma/Documents/clinical-canary/prompts/eval_v1.txt \
  --gt-dir "/home/jma/Documents/clinical-canary/0424_audio_transcripts" \
  --sum-dir "/home/jma/Documents/clinical-canary/0424_summarized_deepseek" \
  --results-dir /home/jma/Documents/clinical-canary/0424_eval_results \
  --prompts-dir /home/jma/Documents/clinical-canary/0424_eval_prompts \
  --api-base https://api.openai.com/v1 \
  --temperature 0
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