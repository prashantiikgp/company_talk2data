# %%
# %% 📝 Souring Path
import sys, os
SRC_PATH = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
print(f"✅ SRC Path: {SRC_PATH}")


# %%
from qdrant_client.http.models import PayloadSchemaType

PAYLOAD_SCHEMA = {
    "company_name": PayloadSchemaType.TEXT,  # TEXT or KEYWORD
    "legal_entity_type": PayloadSchemaType.KEYWORD,
    "state": PayloadSchemaType.KEYWORD,
    "headquarters_city": PayloadSchemaType.KEYWORD,

    "year_founded": {
        "type": PayloadSchemaType.INTEGER,
        "params": {"range": True}
    },

    "company_website": PayloadSchemaType.KEYWORD,
    "logo_url": PayloadSchemaType.KEYWORD,
    "company_description_short": PayloadSchemaType.TEXT,
    "company_description_long": PayloadSchemaType.TEXT,
    "industry_sector": PayloadSchemaType.KEYWORD,

    "total_funding_raised_inr": {
        "type": PayloadSchemaType.INTEGER,
        "params": {"range": True}
    },

    "number_of_funding_rounds": {
        "type": PayloadSchemaType.INTEGER,
        "params": {"range": True}
    },

    "latest_funding_round_type": PayloadSchemaType.KEYWORD,
    "latest_funding_date": PayloadSchemaType.KEYWORD,
    "lead_investors": PayloadSchemaType.KEYWORD,

    "revenue_estimate_annual": {
        "type": PayloadSchemaType.INTEGER,
        "params": {"range": True}
    },

    "valuation_estimate_if_available": {
        "type": PayloadSchemaType.INTEGER,
        "params": {"range": True}
    },

    "number_of_employees_current": {
        "type": PayloadSchemaType.INTEGER,
        "params": {"range": True}
    },

    "number_of_employees_estimate_range": PayloadSchemaType.KEYWORD,
    "key_people": PayloadSchemaType.KEYWORD,
    "founders": PayloadSchemaType.KEYWORD,
    "board_members_advisors": PayloadSchemaType.KEYWORD,

    "employee_growth_yoy": {
        "type": PayloadSchemaType.FLOAT,
        "params": {"range": True}
    },

    "hiring_status": PayloadSchemaType.KEYWORD,
    "popular_roles_open": PayloadSchemaType.KEYWORD,
    "primary_products_services": PayloadSchemaType.KEYWORD,
    "product_categories": PayloadSchemaType.KEYWORD,
    "tech_stack": PayloadSchemaType.KEYWORD,
    "integrations_apis_offered": PayloadSchemaType.KEYWORD,
    "target_market": PayloadSchemaType.KEYWORD,
    "major_customers_logos": PayloadSchemaType.KEYWORD,
    "press_mentions_recent_news": PayloadSchemaType.KEYWORD,
    "competitors": PayloadSchemaType.KEYWORD
}



