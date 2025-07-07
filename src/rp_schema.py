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
        "required": True
    },
    "max_ref_len": {
        "type": int,
        "required": True
    },
    "speed": {
        "type": float,
        "required": False,
    },
    "enhance_audio": {
        "type": bool,
        "required": True
    }
}
