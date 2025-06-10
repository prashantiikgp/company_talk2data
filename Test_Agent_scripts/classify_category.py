# %%
import re
from typing import List, Dict
from langchain_core.tools import Tool 



# 🧩 Expandable category dictionary with synonyms
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "SaaS": ["saas", "software-as-a-service", "cloud software"],
    "B2B": ["b2b", "business to business"],
    "B2C": ["b2c", "business to consumer"],
    "Logistics": ["logistics", "supply chain", "delivery", "transport"],
    "Edtech": ["edtech", "education technology", "online learning"],
    "AgriTech": ["agritech", "agriculture technology", "farming tech"],
    "Fintech": ["fintech", "financial technology", "payments", "digital bank"],
    "HealthTech": ["healthtech", "health tech", "digital health", "telemedicine"],
    "E-commerce": ["e-commerce", "ecommerce", "online store", "digital retail"],
}

def classify_categories(query: str) -> List[str]:
    query_lower = query.lower()
    detected_categories = []

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", query_lower):
                detected_categories.append(category)
                break  # Stop checking more keywords for this category

    return {"industry_category": list(set(detected_categories))}  # Remove duplicates





# %%

# ✅ Example usage:
if __name__ == "__main__":
    query = "Find B2C , b2B or e-commerce startups in the SaaS and logistics space"
    print(classify_categories(query))



