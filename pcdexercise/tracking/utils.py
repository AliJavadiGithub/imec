import re

def extract_timestamp_ms(filename: str) -> int:
    match = re.search(r"(\d+)ms", filename)
    if not match:
        raise ValueError(f"Invalid timestamp in filename: {filename}")
    return int(match.group(1))
