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
        "required": False
    },
    "max_ref_len": {
        "type": int,
        "required": False
    },
    "speed": {
        "type": float,
        "required": False,
    },
    "enhance_audio": {
        "type": bool,
        "required": False
    }
}
