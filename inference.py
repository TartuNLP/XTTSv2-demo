import os
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
from scipy.io.wavfile import write


text = "Tere, tore maailm!"
language = "et"
reference_clip = "LJ001-0030.wav"
file_output = True


TOKENIZER_FILE_LINK = "https://huggingface.co/tartuNLP/XTTS-v2-est/resolve/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://huggingface.co/tartuNLP/XTTS-v2-est/resolve/main/model.pth"
XTTS_CONFIG_LINK = "https://huggingface.co/tartuNLP/XTTS-v2-est/resolve/main/config.json"

MODEL_PATH = "model"
OUTPUT_PATH = "output"
TOKENIZER_FILE = os.path.join(MODEL_PATH, 'vocab.json')
XTTS_CHECKPOINT = os.path.join(MODEL_PATH, 'model.pth')
XTTS_CONFIG_FILE = os.path.join(MODEL_PATH, 'config.json')

# download model files if needed
if not os.path.isfile(TOKENIZER_FILE):
    print(" > Downloading tokenizer!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK], MODEL_PATH, progress_bar=True
    )
if not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading checkpoint!")
    ModelManager._download_model_files(
        [XTTS_CHECKPOINT_LINK], MODEL_PATH, progress_bar=True
    )
if not os.path.isfile(XTTS_CONFIG_FILE):
    print(" > Downloading config!")
    ModelManager._download_model_files(
        [XTTS_CONFIG_LINK], MODEL_PATH, progress_bar=True
    )


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model():
    config = XttsConfig()
    config.load_json(XTTS_CONFIG_FILE)
    XTTS_MODEL = Xtts.init_from_config(config)
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_FILE, use_deepspeed=False)
    XTTS_MODEL.to(device)
    return XTTS_MODEL

model = load_model()

def synthesis(text, language):
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=reference_clip,
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )

    wav_chunks = []
    for chunk in model.inference_stream(
        text=text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.1,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=10,
        top_p=0.3,
    ):
        if chunk is not None:
            wav_chunks.append(chunk)
    
    audio = torch.cat(wav_chunks, dim=0).unsqueeze(0)[0].numpy()

    write(os.path.join(OUTPUT_PATH, text[:15] + '.wav'), 22050, audio)


synthesis(text, language)