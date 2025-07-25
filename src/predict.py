import os
import numpy as np
# torch
import torch
import torch.nn.functional as F
# xtts
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from audio_enhancer import AudioEnhancer

# Constants
SAMPLE_RATE = 24000

use_cuda = os.environ.get('WORKER_USE_CUDA', 'True').lower() == 'true'

def apply_crossfade(wave1, wave2, fade_length_samples=1024):
    """
    Apply crossfade between two audio segments to prevent clicks and pops.
    
    This function overlaps the end of the first segment with the beginning of the second
    segment, applying fade-out to the first and fade-in to the second. Uses square-root
    curves for constant power crossfading, maintaining perceived volume.
    
    Args:
        wave1: First audio segment (torch.Tensor)
        wave2: Second audio segment (torch.Tensor) 
        fade_length_samples: Length of crossfade in samples (default: 1024 ~= 43ms at 24kHz)
                           Recommended: 240-4800 samples (10-200ms at 24kHz)
    
    Returns:
        Concatenated audio with smooth crossfade transition
        
    Note:
        - Shorter crossfades (10-50ms) preserve speech clarity but may still have artifacts
        - Longer crossfades (50-200ms) eliminate artifacts but may cause slight audio blending
        - Default 50ms provides good balance between artifact removal and speech clarity
    """
    if wave1 is None:
        return wave2
    if wave2 is None:
        return wave1
    
    # Ensure both waves are 1D
    wave1 = wave1.squeeze()
    wave2 = wave2.squeeze()
    
    # Validate fade length
    fade_length_samples = max(0, int(fade_length_samples))
    if fade_length_samples == 0:
        return torch.cat([wave1, wave2], dim=0)
    
    # Ensure device and dtype consistency
    if wave1.device != wave2.device:
        wave2 = wave2.to(wave1.device)
    if wave1.dtype != wave2.dtype:
        wave2 = wave2.to(wave1.dtype)
    
    # Limit fade length - don't take more than 50% of either segment to preserve content
    max_fade = min(len(wave1) // 2, len(wave2) // 2)
    fade_length = min(fade_length_samples, max_fade)
    
    if fade_length <= 0:
        # If no overlap possible, just concatenate
        return torch.cat([wave1, wave2], dim=0)
    
    # Create fade curves with matching device and dtype
    fade_out = torch.sqrt(torch.linspace(1.0, 0.0, fade_length, device=wave1.device, dtype=wave1.dtype))
    fade_in = torch.sqrt(torch.linspace(0.0, 1.0, fade_length, device=wave1.device, dtype=wave1.dtype))
    
    # Extract the regions to crossfade
    wave1_fade = wave1[-fade_length:] * fade_out
    wave2_fade = wave2[:fade_length] * fade_in
    
    # Create crossfaded region
    crossfaded = wave1_fade + wave2_fade
    
    # Build result - handle edge cases properly
    parts = []
    if len(wave1) > fade_length:
        parts.append(wave1[:-fade_length])
    parts.append(crossfaded)
    if len(wave2) > fade_length:
        parts.append(wave2[fade_length:])
    
    # Concatenate all parts
    result = torch.cat(parts, dim=0)
    
    return result

def add_silence_with_fade(wave, silence, fade_length_samples=512):
    """
    Add silence to audio with a short fade to prevent clicks.
    
    Args:
        wave: Audio segment
        silence: Silence segment to add
        fade_length_samples: Length of fade in samples (default: 512 ~= 21ms at 24kHz)
    
    Returns:
        Audio with faded silence added
    """
    if wave is None:
        return silence
    if silence is None:
        return wave
    
    wave = wave.squeeze()
    silence = silence.squeeze()
    
    # Validate fade length
    fade_length_samples = max(0, int(fade_length_samples))
    if fade_length_samples == 0:
        return torch.cat([wave, silence], dim=0)
    
    # Ensure device and dtype consistency
    if wave.device != silence.device:
        silence = silence.to(wave.device)
    if wave.dtype != silence.dtype:
        silence = silence.to(wave.dtype)
    
    # Apply short fade-out to the end of the wave to prevent clicks with silence
    # Don't take more than 25% of the wave to preserve content
    max_fade = len(wave) // 4
    fade_length = min(fade_length_samples, max_fade, len(wave))
    
    if fade_length > 0:
        # Fade to 10% not 0% to maintain some presence and avoid complete silence artifacts
        fade_out = torch.sqrt(torch.linspace(1.0, 0.1, fade_length, device=wave.device, dtype=wave.dtype))
        wave = wave.clone()  # Don't modify the original
        wave[-fade_length:] *= fade_out
    
    return torch.cat([wave, silence], dim=0)

def _test_crossfade_functions():
    """Test function to validate crossfade implementation - for development/debugging only."""
    try:
        print("Testing crossfade functions...")
        
        # Test basic crossfade
        wave1 = torch.randn(1000) * 0.5
        wave2 = torch.randn(800) * 0.5
        result = apply_crossfade(wave1, wave2, 100)
        assert result.shape[0] > 0, "Crossfade result should not be empty"
        print(f"✓ Basic crossfade: {wave1.shape} + {wave2.shape} -> {result.shape}")
        
        # Test device consistency
        if use_cuda and torch.cuda.is_available():
            wave1_cuda = wave1.cuda()
            wave2_cpu = wave2.cpu()
            result = apply_crossfade(wave1_cuda, wave2_cpu, 50)
            assert result.device == wave1_cuda.device, "Result should match first wave's device"
            print("✓ Device consistency test passed")
        
        # Test edge cases
        tiny_wave = torch.randn(10)
        normal_wave = torch.randn(1000)
        result = apply_crossfade(tiny_wave, normal_wave, 100)
        assert result.shape[0] > 0, "Edge case result should not be empty"
        print("✓ Edge case test passed")
        
        # Test silence fade
        wave = torch.randn(500) * 0.5
        silence = torch.zeros(200)
        result = add_silence_with_fade(wave, silence, 50)
        assert result.shape[0] == len(wave) + len(silence), "Silence fade should preserve length"
        print("✓ Silence fade test passed")
        
        # Test 0.9 second pause preservation
        speech_segment = torch.randn(int(2.0 * SAMPLE_RATE)) * 0.5  # 2 seconds of speech
        pause_09sec = torch.zeros(int(0.9 * SAMPLE_RATE))  # 0.9 seconds silence
        result = add_silence_with_fade(speech_segment, pause_09sec, int(0.025 * SAMPLE_RATE))  # 25ms fade
        expected_length = len(speech_segment) + len(pause_09sec)
        actual_length = len(result)
        pause_duration_ms = len(pause_09sec) / SAMPLE_RATE * 1000
        print(f"✓ 0.9s pause test: Expected {expected_length} samples, got {actual_length}, pause={pause_duration_ms:.1f}ms")
        assert actual_length == expected_length, f"0.9s pause not preserved: {actual_length} != {expected_length}"
        
        print("All crossfade tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Crossfade test failed: {e}")
        return False

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
        
        # Test crossfade functions to ensure they work correctly
        if not _test_crossfade_functions():
            print("⚠️  Crossfade tests failed - audio may have artifacts")
        else:
            print("✅ Crossfade system ready")

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
            enable_text_splitting: bool = True,
            # Crossfade parameters to prevent clicks/pops
            crossfade_length_ms: float = 50.0,  # Crossfade length in milliseconds
            silence_fade_length_ms: float = 25.0  # Fade length when adding silence
    ):
        silence = torch.zeros(1, int(0.9 * SAMPLE_RATE))
        # Create 0.4 second silence for newline pauses
        newline_silence = torch.zeros(1, int(0.4 * SAMPLE_RATE))
        if use_cuda:
            silence = silence.cuda()
            newline_silence = newline_silence.cuda()
        
        # Validate and convert crossfade lengths from milliseconds to samples
        crossfade_length_ms = max(0.0, min(500.0, float(crossfade_length_ms)))  # Clamp 0-500ms
        silence_fade_length_ms = max(0.0, min(200.0, float(silence_fade_length_ms)))  # Clamp 0-200ms
        
        crossfade_samples = int(crossfade_length_ms * SAMPLE_RATE / 1000.0)
        silence_fade_samples = int(silence_fade_length_ms * SAMPLE_RATE / 1000.0)
        
        print(f"Crossfade settings: {crossfade_length_ms}ms ({crossfade_samples} samples), "
              f"Silence fade: {silence_fade_length_ms}ms ({silence_fade_samples} samples)")
        
        wave, sr = None, None
        
        # Process each text segment
        for line_idx, line in enumerate(text):
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
            
            # Split text content by newlines to create 0.4 sec pauses
            text_segments = text_content.split('\n')
            
            # Clean and improve text segments for better TTS synthesis
            cleaned_segments = []
            for segment in text_segments:
                segment = segment.strip()
                if segment:
                    # Add punctuation if missing to help XTTS with sentence boundaries
                    # Using semicolon + space for natural punctuation spacing
                    if not segment.endswith(('.', '!', '?', ',', ';', ':')):
                        segment += ' ; '
                    cleaned_segments.append(segment)
            
            text_segments = cleaned_segments
            
            for segment_idx, text_segment in enumerate(text_segments):
                # Skip empty segments
                if not text_segment.strip():
                    # If it's an empty segment but not the last one, add newline pause
                    if segment_idx < len(text_segments) - 1:
                        if wave is None:
                            wave = newline_silence.clone()
                            sr = SAMPLE_RATE
                        else:
                            newline_pause = newline_silence.clone()
                            wave = add_silence_with_fade(wave, newline_pause, silence_fade_samples)
                    continue
                
                print(f"Synthesizing: '{text_segment}' with speaker: {speaker_id}")
                
                try:
                    # Synthesize audio for this segment with advanced quality parameters
                    outputs = self.model.synthesize(
                        text_segment,
                        self.config,
                        speaker_wav=voice,
                        gpt_cond_len=gpt_cond_len,
                        gpt_cond_chunk_len=gpt_cond_chunk_len,
                        language=language,
                        max_ref_len=max_ref_len,
                        sound_norm_refs=sound_norm_refs,
                        enable_text_splitting=False,  # Disable internal splitting since we handle newlines manually
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
                        _wave = _wave.squeeze()
                        wave = apply_crossfade(wave, _wave, crossfade_samples)
                    
                    # Add 0.4 sec pause after each text segment (except the last one)
                    if segment_idx < len(text_segments) - 1:
                        wave = wave.squeeze()
                        newline_pause = newline_silence.clone().squeeze()
                        wave = add_silence_with_fade(wave, newline_pause, silence_fade_samples)
                        
                except Exception as e:
                    print(f"Error synthesizing text '{text_segment}': {e}")
                    raise
            
            # Add 0.9 sec silence between different lines in the text list (if there are more lines to process)
            if line_idx < len(text) - 1:
                wave = wave.squeeze()
                silence_to_add = silence.clone().squeeze()
                wave = add_silence_with_fade(wave, silence_to_add, silence_fade_samples)
        
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
