# %%
import re
from typing import Dict, Union
from langchain_core.tools import Tool  


import re
from typing import Dict, Union

def extract_numeric_constraints(query: str) -> Dict[str, Dict[str, Union[int, float]]]:
    constraints = {}

    patterns = [
        (r"(?:over|above|more than|greater than)\s[₹$]?\s?([\d,.]+)([a-zA-Z]*)", "gte"),
        (r"(?:min(?:imum)?|at least)\s[₹$]?\s?([\d,.]+)([a-zA-Z]*)", "gte"),
        (r"\b>[₹$]?\s?([\d,.]+)([a-zA-Z]*)", "gte"),
        (r"(?:under|below|less than|smaller than)\s[₹$]?\s?([\d,.]+)([a-zA-Z]*)", "lte"),
        (r"(?:max(?:imum)?|at most|upto)\s[₹$]?\s?([\d,.]+)([a-zA-Z]*)", "lte"),
        (r"\b<[₹$]?\s?([\d,.]+)([a-zA-Z]*)", "lte"),
        (r"between\s(?:₹|\$)?\s?([\d,.]+)\s?(cr|crore|million|billion|lakhs|lakh)?\s?and\s(?:₹|\$)?\s?([\d,.]+)\s?(cr|crore|million|billion|lakhs|lakh)?", "range"),

    ]

    field_keywords = {
        "total_funding_raised_inr": ["funding", "investment", "raised", "capital", "secured"],
        "valuation_estimate_if_available": ["valuation", "worth", "company value"],
        "revenue_estimate_annual": ["revenue", "income", "sales", "turnover"],
        "number_of_employees_current": ["employees", "staff", "team size", "headcount"],
        "year_founded": ["founded", "established", "startup year", "launch year"]
    }

    query_lower = query.lower()

    def normalize_unit(value: str, unit: str) -> float:
        num = float(value.replace(",", "").strip())
        unit = unit.lower()
        if unit in ["million", "m"]:
            return round(num * 0.1, 2)
        elif unit in ["billion", "b"]:
            return round(num * 100, 2)
        elif unit in ["lakh", "lakhs", "l"]:
            return round(num * 0.1, 2)
        elif unit in ["cr", "crore", "crores", "c"]:
            return round(num, 2)
        else:
            return round(num, 2)  # Default to crore

    for pattern, operator in patterns:
        for match in re.finditer(pattern, query_lower):
            #print(f"\n[Pattern Match] Operator: {operator}, Match: '{match.group()}', Span: {match.start()}-{match.end()}")
            span_start = match.start()
            matched_field = None
            nearest_keyword = float("inf")

            if operator == "range":
                val1, unit1, val2, unit2 = match.groups()
                low = normalize_unit(val1, unit1 or "cr")
                high = normalize_unit(val2, unit2 or "cr")
               # print(f"→ Extracted range: ₹{low} to ₹{high}")
            else:
                val, unit = match.groups()
                val = normalize_unit(val, unit or "cr")
                #print(f"→ Extracted value: ₹{val} with operator '{operator}'")

            #print(f"[Keyword Search] Searching for closest field near index {span_start}...")

            for field, keywords in field_keywords.items():
                for keyword in keywords:
                    for m_kw in re.finditer(re.escape(keyword), query_lower, flags=re.IGNORECASE):
                        distance = abs(m_kw.start() - span_start)
                        #print(f"  Checking keyword '{keyword}' for field '{field}' at index {m_kw.start()} → distance {distance}")
                        if distance < nearest_keyword and distance <= 50:
                            nearest_keyword = distance
                            matched_field = field

            if not matched_field:
                matched_field = "total_funding_raised_inr"
                print("⚠️ No matching field found nearby. Falling back to 'total_funding_raised_inr'.")

            #print(f"✅ Matched field: {matched_field}")

            if operator == "range":
                constraints.setdefault(matched_field, {}).update({
                    "gte": low,
                    "lte": high
                })
            else:
                constraints.setdefault(matched_field, {}).update({
                    operator: val
                })

    # 📆 Year-specific fallback extraction
    year_after = re.search(r"(?:after|since)\s(\d{4})", query_lower)
    if year_after:
        print(f"\n[Year Constraint] Found 'after {year_after.group(1)}'")
        constraints.setdefault("year_founded", {})["gte"] = int(year_after.group(1))

    year_before = re.search(r"(?:before|until|prior to)\s(\d{4})", query_lower)
    if year_before:
        print(f"[Year Constraint] Found 'before {year_before.group(1)}'")
        constraints.setdefault("year_founded", {})["lte"] = int(year_before.group(1))

    return constraints





# %%
# ✅ Example usage:
if __name__ == "__main__":
    query = "tell me companies with funding over 10 million and revenue less than 5 crore, founded after 2010"
    print(extract_numeric_constraints(query))



