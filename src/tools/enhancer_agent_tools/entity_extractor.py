# %%
# ------------------------------------
# Tool 4: Entity Extractor
# ------------------------------------
# 🔍 Predefined Lists (Can be loaded from DB or external sources)

import re
from typing import List, Dict
from langchain_core.tools import Tool  

CITIES = [
    "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Gurgaon",
    "Noida", "Ahmedabad", "Kolkata", "Jaipur", "Surat", "Lucknow", "Indore"
]

STATES = [
    "Karnataka", "Maharashtra", "Tamil Nadu", "Delhi", "Telangana",
    "Uttar Pradesh", "Rajasthan", "Gujarat", "West Bengal", "Madhya Pradesh"
]

INDUSTRIES = [
    "SaaS", "Fintech", "HealthTech", "Edtech", "Logistics", "AgriTech",
    "E-commerce", "AI", "CleanTech", "Gaming", "FoodTech", "PropTech",
    "LegalTech", "IoT", "Cybersecurity", "TravelTech", "MedTech"
]

INVESTOR_TYPES = [
    "Angel", "VC", "Venture Capital", "Private Equity", "Accelerator",
    "Incubator", "Corporate", "Family Office", "Crowdfunding Platform"
]

FUNDING_ROUND_TYPES = [
    "Pre-seed", "Seed", "Angel", "Series A", "Series B", "Series C",
    "Series D", "IPO", "Debt Financing", "Bridge Round", "Grants"
]

PRODUCT_CATEGORIES = [
    "B2B", "B2C", "D2C", "Marketplace", "Subscription", "Enterprise SaaS",
    "API Platform", "Mobile App", "Hardware", "Cloud Services"
]

HIRING_STATUSES = [
    "Hiring", "Actively Hiring", "Not Hiring", "Hiring Freeze", "Remote Hiring"
]

# 🔧 Synonym / Variant Normalization Map

SYNONYM_MAP = {
    "bangalore": "Bengaluru",
    "b'lore": "Bengaluru",
    "madras": "Chennai",
    "delhi ncr": "Delhi",
    "ncr": "Delhi",
    "mum": "Mumbai",
    "mumbai metropolitan": "Mumbai",
    "hydrabad": "Hyderabad",
    "noida extension": "Noida",
    "gurugram": "Gurgaon"
}


def normalize_text(text: str) -> str:
    text = text.lower()
    for syn, canonical in SYNONYM_MAP.items():
        pattern = r'\b' + re.escape(syn.lower()) + r'\b'
        text = re.sub(pattern, canonical.lower(), text)
    return text


def extract_entities(query: str) -> Dict[str, List[str]]:
    query = normalize_text(query)
    found = {
        "cities": [],
        "states": [],
        "industries": [],
        "investor_types": [],
        "funding_round_types": [],
        "product_categories": [],
        "hiring_statuses": []
    }

    def match_entities(entity_list, label):
        for item in entity_list:
            if re.search(rf"\b{item.lower()}\b", query):
                found[label].append(item)

    match_entities(CITIES, "cities")
    match_entities(STATES, "states")
    match_entities(INDUSTRIES, "industries")
    match_entities(INVESTOR_TYPES, "investor_types")
    match_entities(FUNDING_ROUND_TYPES, "funding_round_types")
    match_entities(PRODUCT_CATEGORIES, "product_categories")
    match_entities(HIRING_STATUSES, "hiring_statuses")

    return found





# %%
# ✅ Example usage:
if __name__ == "__main__":
    query = "give me bengaluru based startups in fintech and saas, also check if there are any hiring opportunities in delhi ncr or mumbai"
    print(extract_entities(query))



