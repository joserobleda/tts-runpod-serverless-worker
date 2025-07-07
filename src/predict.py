import os
import numpy as np
# torch
import torch
# xtts
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from audio_enhancer import AudioEnhancer

# Constants
SAMPLE_RATE = 24000

use_cuda = os.environ.get('WORKER_USE_CUDA', 'True').lower() == 'true'

class Predictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def setup(self):
        # Load XTTSv2 model
        self.config = XttsConfig()
        self.config.load_json(
            os.path.join(self.model_dir, "xttsv2", "config.json")
        )
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config,
            checkpoint_dir=os.path.join(self.model_dir, "xttsv2"),
            use_deepspeed=True,
            eval=True
        )
        if use_cuda:
            self.model.cuda()
        # Load Audio Enhancer model
        self.audio_enhancer = AudioEnhancer.from_pretrained(
            os.path.join(self.model_dir, "audio_enhancer", "enhancer_stage2"),
            "cuda" if use_cuda else "cpu"
        )

    @torch.inference_mode()
    def predict(
            self,
            text: list,
            speaker_wav: dict,
            gpt_cond_len: int,
            max_ref_len: int,
            language: str,
            speed: float,
            enhance_audio: bool
    ):
        silence = torch.zeros(1, int(0.10 * SAMPLE_RATE))
        if use_cuda:
            silence = silence.cuda()
        
        wave, sr = None, None
        
        # Process each text segment
        for line in text:
            # Handle different input formats
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                # Format: [speaker_id, text_content]
                speaker_id, text_content = line[0], line[1]
            elif isinstance(line, dict):
                # Format: {"speaker": "id", "text": "content"}
                speaker_id = line.get("speaker", list(speaker_wav.keys())[0])
                text_content = line.get("text", "")
            elif isinstance(line, str):
                # Format: plain text string, use first available speaker
                speaker_id = list(speaker_wav.keys())[0]
                text_content = line
            else:
                continue
            
            # Get the voice file for this speaker
            voice = speaker_wav.get(speaker_id)
            if voice is None:
                # Fallback to first available voice
                voice = list(speaker_wav.values())[0]
            
            # Synthesize audio for this segment
            outputs = self.model.synthesize(
                text_content,
                self.config,
                speaker_wav=voice,
                gpt_cond_len=gpt_cond_len,
                language=language,
                enable_text_splitting=True,
                max_ref_len=max_ref_len,
                speed=speed
            )
            
            _wave, _sr = outputs['wav'], SAMPLE_RATE
            
            # Concatenate audio segments
            if wave is None:
                wave = _wave
                sr = _sr
            else:
                wave = torch.cat([wave, silence.clone(), _wave], dim=1)
        
        # Enhance audio if requested
        if enhance_audio and wave is not None:
            try:
                print(f"Enhancing audio: input shape={wave.shape}, sr={sr}")
                # wave is already a torch tensor, so don't convert from numpy
                enhanced_wave, enhanced_sr = self.audio_enhancer(wave, sr)
                wave = enhanced_wave
                sr = enhanced_sr
                print(f"Audio enhanced: output shape={wave.shape}, sr={sr}")
            except Exception as e:
                print(f"Audio enhancement failed: {e}, using original audio")
                # Continue with original audio if enhancement fails
        
        # Convert to numpy for return
        if wave is not None:
            wave = wave.detach().cpu().numpy()
        
        return wave, sr
