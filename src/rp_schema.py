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
    }
}
