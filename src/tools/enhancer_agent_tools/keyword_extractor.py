##--Keyword Extractor--##
# This script extracts keywords from a given query string and maps them to specific fields.
# It uses a predefined dictionary of filterable fields and their associated keywords.
# The script defines a function `keyword_extractor_fn` that takes a query string as input.

import re
from typing import Dict, Any
from langchain_core.tools import Tool

FILTERABLE_FIELDS: Dict[str, list] = {
    # 🌍 Location
    "headquarters_city": [
        "city", "headquarters", "based in", "location", "bengaluru", "bangalore",
        "mumbai", "delhi", "chennai", "kolkata", "hyderabad", "jaipur", "ahmedabad"
    ],
    "state": [
        "state", "region", "area", "in maharashtra", "in karnataka", "in gujarat"
    ],

    # 🏭 Industry + Category (merged with classifier synonyms)
    "industry_sector": [
        "industry", "sector", "domain", "vertical",
        "saas", "software-as-a-service", "cloud software",
        "b2b", "business to business",
        "b2c", "business to consumer",
        "d2c", "direct to consumer",
        "fintech", "financial technology", "digital bank", "payments",
        "ecommerce", "online store", "digital retail",
        "logistics", "supply chain", "delivery", "transport",
        "healthtech", "digital health", "telemedicine",
        "edtech", "education technology", "online learning",
        "agritech", "agriculture technology", "farming tech",
        "cleantech", "climate tech", "sustainability"
    ],

    # 💰 Financial
    "total_funding_raised_inr": [
        "funding", "raised", "capital", "total raised", "total investment", "₹", "cr",
        "crore", "million", "billion", "over ₹", "less than ₹", "above ₹", "under ₹"
    ],
    "valuation_estimate_if_available": [
        "valuation", "worth", "company valuation", "valued at", "estimated worth",
        "₹ valuation", "how much is it worth"
    ],
    "revenue_estimate_annual": [
        "revenue", "income", "turnover", "sales", "₹ revenue", "annual revenue",
        "earning"
    ],
    "number_of_funding_rounds": [
        "funding rounds", "number of rounds", "how many times", "rounds raised"
    ],
    "latest_funding_round_type": [
        "round type", "series a", "series b", "seed", "pre-seed", "bridge round",
        "angel", "growth round", "venture round"
    ],
    "latest_funding_date": [
        "funding date", "latest funding", "last raised", "when did it raise"
    ],

    # 👥 People
    "founders": [
        "founders", "co-founders", "started by", "founded by", "entrepreneurs",
        "startup founders", "iit", "iim", "alumni", "serial entrepreneur"
    ],
    "board_members__advisors": [
        "board", "advisors", "board members", "mentors", "director"
    ],

    # 🧑‍💼 Hiring
    "hiring_status": [
        "hiring", "actively hiring", "hiring freeze", "currently hiring", "not hiring"
    ],
    "popular_roles_open": [
        "roles", "job openings", "positions", "jobs", "vacancies", "engineers",
        "product managers", "sales roles", "open roles"
    ],

    # 📦 Products / Tech
    "primary_products__services": [
        "products", "services", "offers", "tools", "solutions", "platform",
        "mobile app", "AI tools", "API", "product line", "dashboard", "crm", "analytics"
    ],
    "product_categories": [
        "category", "type of product", "business model", "B2B", "B2C", "D2C", "SMB",
        "enterprise", "consumer", "retail", "wholesale"
    ],
    "tech_stack": [
        "technology", "tech stack", "framework", "platform", "built on", "python",
        "node.js", "aws", "gcp", "react", "java", "spring boot", "azure"
    ],
    "integrations__apis_offered": [
        "integrates with", "API", "integrations", "third-party tools", "razorpay",
        "tally", "zoho", "crm"
    ],

    # 📣 Market Presence
    "major_customers__logos": [
        "clients", "customers", "buyers", "logos", "key accounts", "target accounts",
        "enterprise clients", "big customers"
    ],
    "competitors": [
        "competitors", "similar to", "like", "versus", "against", "same space",
        "competing with", "similar company"
    ],

    # 📆 Time & Growth
    "year_founded": [
        "founded", "established", "founded in", "year of founding", "launched",
        "started in", "beginning"
    ],
    "employee_growth_yoy_": [
        "employee growth", "hiring trend", "growth rate", "yoy growth", "team growth"
    ],
    "number_of_employees_current": [
        "employees", "team size", "staff", "how many people", "employee count",
        "current employees"
    ],
    "number_of_employees_estimate_range": [
        "employee range", "size of team", "headcount range"
    ],

    # 🆕 Named Entities & Tags
    "company_name": ["flipkart", "paytm", "zomato", "cred", "byjus", "zoho", "freshworks"],
    "investors": ["sequoia", "accel", "tiger global", "blume ventures", "softbank", "matrix partners"],
    "labels": ["unicorn", "soonicorn", "top startup", "high growth", "bootstrap", "market leader"]
}


def keyword_extractor_fn(query: str) -> Dict[str, str]:
    query_lower = query.lower()
    result: Dict[str, Any] = {}

    for field, keywords in FILTERABLE_FIELDS.items():
        matched = [kw for kw in keywords if kw in query_lower]
        if matched:
            result[field] = list(set(matched))

    return result

# %%
# ✅ Example usage:
if __name__ == "__main__":
    query = "tell me companies with funding over 10 million and revenue less than 5 crore, founded after 2010"
    print(keyword_extractor_fn(query))


