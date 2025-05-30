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

def compose_filters(*tools_outputs: str) -> Dict[str, Any]:
    final_filters = {}

    for tool_output_str in tools_outputs:
        try:
            tool_output = json.loads(tool_output_str)
            if not isinstance(tool_output, dict):
                continue
        except json.JSONDecodeError:
            continue  # skip if it's not valid JSON

        for key, value in tool_output.items():
            if not value:
                continue
            if key in final_filters:
                final_filters[key] = merge_values(final_filters[key], value)
            else:
                final_filters[key] = value

    return final_filters



# %%




# %%



