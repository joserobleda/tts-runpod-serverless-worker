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
        print("Loading XTTS model...")
        try:
            # Load XTTSv2 model
            self.config = XttsConfig()
            config_path = os.path.join(self.model_dir, "xttsv2", "config.json")
            print(f"Loading config from: {config_path}")
            self.config.load_json(config_path)
            
            self.model = Xtts.init_from_config(self.config)
            checkpoint_dir = os.path.join(self.model_dir, "xttsv2")
            print(f"Loading checkpoint from: {checkpoint_dir}")
            
            # Load without DeepSpeed for compatibility
            self.model.load_checkpoint(
                self.config,
                checkpoint_dir=checkpoint_dir,
                use_deepspeed=False,  # Disabled for compatibility
                eval=True
            )
            
            if use_cuda:
                print("Moving model to CUDA...")
                self.model.cuda()
            
            print("XTTS model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading XTTS model: {e}")
            raise
        
        # Load Audio Enhancer model
        try:
            print("Loading audio enhancer...")
            enhancer_path = os.path.join(self.model_dir, "audio_enhancer", "enhancer_stage2")
            print(f"Loading enhancer from: {enhancer_path}")
            self.audio_enhancer = AudioEnhancer.from_pretrained(
                enhancer_path,
                "cuda" if use_cuda else "cpu"
            )
            print("Audio enhancer loaded successfully!")
        except Exception as e:
            print(f"Error loading audio enhancer: {e}")
            print("Continuing without audio enhancer...")
            self.audio_enhancer = None

    @torch.inference_mode()
    def predict(
            self,
            text: list,
            speaker_wav: dict,
            gpt_cond_len: int,
            max_ref_len: int,
            language: str,
            speed: float,
            enhance_audio: bool,
            # Advanced quality parameters
            temperature: float = 0.7,
            length_penalty: float = 1.0,
            repetition_penalty: float = 5.0,
            top_k: int = 50,
            top_p: float = 0.8,
            num_gpt_outputs: int = 1,
            gpt_cond_chunk_len: int = 4,
            sound_norm_refs: bool = False,
            enable_text_splitting: bool = True
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
            
            print(f"Synthesizing: '{text_content}' with speaker: {speaker_id}")
            
            try:
                # Synthesize audio for this segment with advanced quality parameters
                outputs = self.model.synthesize(
                    text_content,
                    self.config,
                    speaker_wav=voice,
                    gpt_cond_len=gpt_cond_len,
                    gpt_cond_chunk_len=gpt_cond_chunk_len,
                    language=language,
                    max_ref_len=max_ref_len,
                    sound_norm_refs=sound_norm_refs,
                    enable_text_splitting=enable_text_splitting,
                    # Advanced quality parameters
                    temperature=temperature,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    top_k=top_k,
                    top_p=top_p,
                    speed=speed
                )
                
                _wave, _sr = outputs['wav'], SAMPLE_RATE
                
                # Ensure _wave is a torch.Tensor
                if isinstance(_wave, np.ndarray):
                    _wave = torch.from_numpy(_wave)
                    if use_cuda:
                        _wave = _wave.cuda()
                elif not isinstance(_wave, torch.Tensor):
                    _wave = torch.tensor(_wave)
                    if use_cuda:
                        _wave = _wave.cuda()
                
                print(f"Generated audio segment: shape={_wave.shape}, sr={_sr}")
                
                # Concatenate audio segments
                if wave is None:
                    wave = _wave
                    sr = _sr
                else:
                    wave = wave.squeeze()
                    silence_to_add = silence.clone().squeeze()
                    _wave = _wave.squeeze()
                    wave = torch.cat([wave, silence_to_add, _wave], dim=0)
                    
            except Exception as e:
                print(f"Error synthesizing text '{text_content}': {e}")
                raise
        
        # Enhance audio if requested and enhancer is available
        if enhance_audio and wave is not None and self.audio_enhancer is not None:
            try:
                print(f"Enhancing audio: input shape={wave.shape}, sr={sr}, type={type(wave)}")
                
                # Ensure wave is a PyTorch tensor
                if isinstance(wave, np.ndarray):
                    print("Converting numpy array to tensor for enhancement")
                    wave = torch.from_numpy(wave)
                    if use_cuda:
                        wave = wave.cuda()
                elif not isinstance(wave, torch.Tensor):
                    print(f"Unexpected wave type: {type(wave)}, converting to tensor")
                    wave = torch.tensor(wave)
                    if use_cuda:
                        wave = wave.cuda()
                
                # Ensure correct tensor shape (audio enhancer might expect specific dimensions)
                if wave.dim() == 1:
                    # Add batch dimension if needed
                    wave = wave.unsqueeze(0)
                
                print(f"Input to enhancer: shape={wave.shape}, device={wave.device}, dtype={wave.dtype}")
                
                enhanced_wave, enhanced_sr = self.audio_enhancer(wave, sr)
                wave = enhanced_wave
                sr = enhanced_sr
                print(f"Audio enhanced: output shape={wave.shape}, sr={sr}")
                
            except Exception as e:
                print(f"Audio enhancement failed: {e}, using original audio")
                print(f"Wave type: {type(wave)}, shape: {wave.shape if hasattr(wave, 'shape') else 'no shape'}")
                import traceback
                traceback.print_exc()
                # Continue with original audio if enhancement fails
        elif enhance_audio and self.audio_enhancer is None:
            print("Audio enhancement requested but enhancer not available")
        
        # Convert to numpy for return
        if wave is not None:
            if isinstance(wave, torch.Tensor):
                wave = wave.detach().cpu().numpy()
            # Ensure proper shape for output
            if wave.ndim > 1:
                wave = wave.squeeze()
        
        return wave, sr
