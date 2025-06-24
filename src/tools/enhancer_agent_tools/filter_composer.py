# %%

# ------------------------------------
# Tool 5: Filter Composer
# ------------------------------------
import json
from typing import Any, Dict

def merge_values(existing, new):
    if isinstance(existing, dict) and isinstance(new, dict):
        return {**existing, **new}
    elif isinstance(existing, list) and isinstance(new, list):
        return list(set(existing + new))
    elif isinstance(existing, list):
        return list(set(existing + [new]))
    elif isinstance(new, list):
        return list(set([existing] + new))
    elif existing != new:
        return [existing, new]
    else:
        return existing

def compose_filters(*filter_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple filter-dicts into one.

    All arguments **must** be dictionaries. If a non-dict sneaks in, raise
    a clear error so the bug is obvious immediately.
    """
    merged: Dict[str, Any] = {}

    for idx, d in enumerate(filter_dicts, start=1):
        if not isinstance(d, dict):
            raise TypeError(
                f"compose_filters argument #{idx} is {type(d).__name__}; "
                "expected a dict. Check upstream tool outputs."
            )
        # simple shallow merge: later dicts overwrite earlier keys
        merged.update(d)

    return merged


# %%




# %%



