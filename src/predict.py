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


def _apply_overrides(
    current: dict,
    overrides: dict | None,
):
    """Apply absolute or relative overrides to current params in place."""
    if not overrides:
        return current

    # Absolute
    if 'temperature' in overrides:
        current['temperature'] = float(overrides['temperature'])
    if 'top_p' in overrides:
        current['top_p'] = float(overrides['top_p'])
    if 'speed' in overrides:
        current['speed'] = float(overrides['speed'])
    if 'top_k' in overrides:
        current['top_k'] = int(overrides['top_k'])
    if 'length_penalty' in overrides:
        current['length_penalty'] = float(overrides['length_penalty'])
    if 'repetition_penalty' in overrides:
        current['repetition_penalty'] = float(overrides['repetition_penalty'])
    if 'gpt_cond_len' in overrides:
        current['gpt_cond_len'] = int(overrides['gpt_cond_len'])
    if 'gpt_cond_chunk_len' in overrides:
        current['gpt_cond_chunk_len'] = int(overrides['gpt_cond_chunk_len'])
    if 'sound_norm_refs' in overrides:
        current['sound_norm_refs'] = bool(overrides['sound_norm_refs'])
    if 'enable_text_splitting' in overrides:
        current['enable_text_splitting'] = bool(overrides['enable_text_splitting'])

    # Relative
    if 'temperature_delta' in overrides and 'temperature' not in overrides:
        current['temperature'] = float(current['temperature']) + float(overrides['temperature_delta'])
    if 'top_p_delta' in overrides and 'top_p' not in overrides:
        current['top_p'] = float(current['top_p']) + float(overrides['top_p_delta'])
    if 'speed_multiplier' in overrides and 'speed' not in overrides:
        current['speed'] = float(current['speed']) * float(overrides['speed_multiplier'])

    return current


def _clamp_params(p: dict):
    """Clamp parameter dict to safe ranges, mutate and return it."""
    p['temperature'] = max(0.0, min(1.0, float(p['temperature'])))
    p['top_p'] = max(0.0, min(0.98, float(p['top_p'])))
    p['top_k'] = max(1, int(p['top_k']))
    p['speed'] = max(0.5, min(2.0, float(p['speed'])))
    return p


def _build_segment_params(text_segment: str, global_options: dict | None, segment_options: dict | None) -> dict:
    """Compose final parameter set for a segment from defaults, punctuation defaults, global and segment overrides."""
    _go = global_options or {}

    # 1) Defaults
    params = {
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 50,
        'speed': 1.0,
        'length_penalty': 1.0,
        'repetition_penalty': 5.0,
        'gpt_cond_len': 30,
        'gpt_cond_chunk_len': 4,
        'max_ref_len': 60,
        'sound_norm_refs': False,
        'enable_text_splitting': True
    }

    # 2) Punctuation-based defaults (segment overrides > global)
    stripped = text_segment.strip()
    # Defaults for questions: temperature +0.3, top_p +0.4, speed -0.1 (≈ multiplier 0.9)
    q_temp_delta = float((segment_options.get('question_temperature_delta') if segment_options and 'question_temperature_delta' in segment_options else _go.get('question_temperature_delta', 0.30)))
    q_top_p_delta = float((segment_options.get('question_top_p_delta') if segment_options and 'question_top_p_delta' in segment_options else _go.get('question_top_p_delta', 0.40)))
    q_speed_mul = float((segment_options.get('question_speed_multiplier') if segment_options and 'question_speed_multiplier' in segment_options else _go.get('question_speed_multiplier', 0.90)))
    # Defaults for exclamations: temperature +0.3, top_p +0.4, speed -0.1 (≈ multiplier 0.9)
    e_temp_delta = float((segment_options.get('exclamation_temperature_delta') if segment_options and 'exclamation_temperature_delta' in segment_options else _go.get('exclamation_temperature_delta', 0.30)))
    e_top_p_delta = float((segment_options.get('exclamation_top_p_delta') if segment_options and 'exclamation_top_p_delta' in segment_options else _go.get('exclamation_top_p_delta', 0.40)))
    e_speed_mul = float((segment_options.get('exclamation_speed_multiplier') if segment_options and 'exclamation_speed_multiplier' in segment_options else _go.get('exclamation_speed_multiplier', 1.00)))

    if stripped.endswith('?'):
        params['temperature'] = min(1.0, params['temperature'] + max(0.0, q_temp_delta))
        params['top_p'] = min(0.98, params['top_p'] + max(0.0, q_top_p_delta))
        params['speed'] = min(1.2, params['speed'] * max(0.5, min(2.0, q_speed_mul)))
    elif stripped.endswith('!'):
        params['temperature'] = min(1.0, params['temperature'] + max(0.0, e_temp_delta))
        params['top_p'] = min(0.98, params['top_p'] + max(0.0, e_top_p_delta))
        params['speed'] = min(1.2, params['speed'] * max(0.5, min(2.0, e_speed_mul)))

    # 3) Global overrides
    params = _apply_overrides(params, _go)
    # 4) Segment overrides (take precedence)
    params = _apply_overrides(params, segment_options)
    # 5) Clamp
    return _clamp_params(params)

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
            language: str,
            global_options: dict = None
    ):
        silence = torch.zeros(1, int(0.9 * SAMPLE_RATE))
        # Create 0.4 second silence for newline pauses
        newline_silence = torch.zeros(1, int(0.4 * SAMPLE_RATE))
        if use_cuda:
            silence = silence.cuda()
            newline_silence = newline_silence.cuda()
        
        # Read processing options from global options with safe defaults
        _go = global_options or {}
        crossfade_length_ms = max(0.0, min(500.0, float(_go.get('crossfade_length_ms', 50.0))))
        silence_fade_length_ms = max(0.0, min(200.0, float(_go.get('silence_fade_length_ms', 25.0))))
        enhance_audio_flag = bool(_go.get('enhance_audio', True))
        
        crossfade_samples = int(crossfade_length_ms * SAMPLE_RATE / 1000.0)
        silence_fade_samples = int(silence_fade_length_ms * SAMPLE_RATE / 1000.0)
        
        print(f"Crossfade settings: {crossfade_length_ms}ms ({crossfade_samples} samples), "
              f"Silence fade: {silence_fade_length_ms}ms ({silence_fade_samples} samples)")
        
        wave, sr = None, None
        
        # Process each text segment
        for line_idx, line in enumerate(text):
            # Handle different input formats
            segment_options = {}
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                # Format: [speaker_id, text_content, {options}?]
                speaker_id, text_content = line[0], line[1]
                if len(line) >= 3 and isinstance(line[2], dict):
                    segment_options = line[2]
            elif isinstance(line, dict):
                # Format: {"speaker": "id", "text": "content", "options": {...}?}
                speaker_id = line.get("speaker", list(speaker_wav.keys())[0])
                text_content = line.get("text", "")
                segment_options = line.get("options", {}) if isinstance(line.get("options", {}), dict) else {}
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
                if not segment:
                    continue
                # Ensure proper terminal punctuation to help prosody
                if not segment.endswith(('.', '!', '?', ',', ';', ':')):
                    segment = segment + ' ; '
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
                    params = _build_segment_params(text_segment, global_options, segment_options)

                    # Synthesize audio for this segment with advanced quality parameters
                    outputs = self.model.synthesize(
                        text_segment,
                        self.config,
                        speaker_wav=voice,
                        gpt_cond_len=params['gpt_cond_len'],
                        gpt_cond_chunk_len=params['gpt_cond_chunk_len'],
                        language=language,
                        max_ref_len=params['max_ref_len'],
                        sound_norm_refs=params['sound_norm_refs'],
                        enable_text_splitting=params['enable_text_splitting'],
                        # Advanced quality parameters
                        temperature=params['temperature'],
                        length_penalty=params['length_penalty'],
                        repetition_penalty=params['repetition_penalty'],
                        top_k=params['top_k'],
                        top_p=params['top_p'],
                        speed=params['speed']
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
        if enhance_audio_flag and wave is not None and self.audio_enhancer is not None:
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
        elif enhance_audio_flag and self.audio_enhancer is None:
            print("Audio enhancement requested but enhancer not available")
        
        # Convert to numpy for return
        if wave is not None:
            if isinstance(wave, torch.Tensor):
                wave = wave.detach().cpu().numpy()
            # Ensure proper shape for output
            if wave.ndim > 1:
                wave = wave.squeeze()
        
        return wave, sr
