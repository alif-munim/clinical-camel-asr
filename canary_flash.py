from nemo.collections.asr.models import EncDecMultiTaskModel
# load model
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-flash')
# update decode params
decode_cfg = canary_model.cfg.decoding
decode_cfg.beam.beam_size = 1
canary_model.change_decoding_strategy(decode_cfg)

# /home/jma/Documents/vita/scripts/data --> audio files
# /home/jma/Documents/vita/primock57/notes --> ground truth

output = canary_model.transcribe(
    ['/home/jma/Documents/vita/scripts/data/day1_consultation01_patient.wav'],
    batch_size=32,  # batch size to run the inference with
    pnc='yes',        # generate output with Punctuation and Capitalization
)

predicted_text_1 = output
print(predicted_text_1)

