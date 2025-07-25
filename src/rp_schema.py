INPUT_SCHEMA = {
    "language": {
        "type": str,
        "required": True
    },
    "voice": {
        "type": dict,
        "required": True
    },
    "text": {
        "type": list,
        "required": True
    },
    "gpt_cond_len": {
        "type": int,
        "required": False,
        "default": 30
    },
    "max_ref_len": {
        "type": int,
        "required": False,
        "default": 60
    },
    "speed": {
        "type": float,
        "required": False,
        "default": 1.0
    },
    "enhance_audio": {
        "type": bool,
        "required": False,
        "default": True
    },
    "temperature": {
        "type": float,
        "required": False,
        "default": 0.7
    },
    "length_penalty": {
        "type": float,
        "required": False,
        "default": 1.0
    },
    "repetition_penalty": {
        "type": float,
        "required": False,
        "default": 5.0
    },
    "top_k": {
        "type": int,
        "required": False,
        "default": 50
    },
    "top_p": {
        "type": float,
        "required": False,
        "default": 0.8
    },
    "num_gpt_outputs": {
        "type": int,
        "required": False,
        "default": 1
    },
    "gpt_cond_chunk_len": {
        "type": int,
        "required": False,
        "default": 4
    },
    "sound_norm_refs": {
        "type": bool,
        "required": False,
        "default": False
    },
    "enable_text_splitting": {
        "type": bool,
        "required": False,
        "default": True
    },
    "crossfade_length_ms": {
        "type": float,
        "required": False,
        "default": 50.0,
        "min": 0.0,
        "max": 500.0,
        "description": "Crossfade length in milliseconds between audio segments to prevent clicks/pops. Range: 0-500ms. Recommended: 10-200ms for speech."
    },
    "silence_fade_length_ms": {
        "type": float,
        "required": False,
        "default": 25.0,
        "min": 0.0,
        "max": 200.0,
        "description": "Fade length in milliseconds when adding silence to prevent clicks/pops. Range: 0-200ms. Recommended: 5-100ms for speech."
    }
}
