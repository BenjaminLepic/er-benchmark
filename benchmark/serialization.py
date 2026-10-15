import json


SERIALIZATION_MODES = {
    "values_space": "Values space",
    "labels_inline": "Labels inline",
    "labels_multiline": "Labels multiline",
    "labels_values_multiline": "Column value multiline",
    "json_compact": "JSON compact",
    "values_inline": "Values inline",
    "values_multiline": "Values multiline",
}


def serialize_record(fields, mode):
    filtered = {
        key: value
        for key, value in fields.items()
        if value is not None and str(value).strip() != ""
    }
    if not filtered:
        filtered = fields

    if mode == "values_space":
        return " ".join(str(value) for value in filtered.values()).strip()
    if mode == "labels_inline":
        return ", ".join(f"{key}: {value}" for key, value in filtered.items())
    if mode == "labels_multiline":
        return "\n".join(f"{key}: {value}" for key, value in filtered.items())
    if mode == "labels_values_multiline":
        return "\n".join(f"{key} {value}" for key, value in filtered.items())
    if mode == "json_compact":
        return json.dumps(filtered, ensure_ascii=False, separators=(",", ":"))
    if mode == "values_inline":
        return ", ".join(str(value) for value in filtered.values())
    if mode == "values_multiline":
        return "\n".join(str(value) for value in filtered.values())
    raise ValueError(f"Unknown serialization mode: {mode}")
