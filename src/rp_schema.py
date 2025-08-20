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
    "options": {
        "type": dict,
        "required": False,
        "default": {}
    }
}
