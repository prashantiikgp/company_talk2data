import os
import sys

def get_base_dir(levels_up: int = 1) -> str:
    """
    Returns the base directory of the project.

    - Uses __file__ when run from .py
    - Uses os.getcwd() when in notebook
    """
    try:
        # In .py files
        current_file = os.path.abspath(__file__)
        return os.path.abspath(os.path.join(os.path.dirname(current_file), *[".."] * levels_up))
    except NameError:
        # In Jupyter notebooks
        return os.path.abspath(os.path.join(os.getcwd(), *[".."] * levels_up))


def get_vector_store_path(subfolder: str = "faiss_full_row_index") -> str:
    """
    Returns the full path to the FAISS vector store directory.
    """
    base_dir = get_base_dir()
    return os.path.join(base_dir, "database", "vector_store", subfolder)


def get_graph_store_path() -> str:
    """
    Returns path to Neo4j CSVs or configs if needed.
    """
    base_dir = get_base_dir()
    return os.path.join(base_dir, "database", "graph")

def get_qdrant_store_path(subfolder: str = "collection") -> str:
    base_dir = get_base_dir()
    return os.path.join(base_dir, "database", "qdrant_store_local_db", subfolder)

print(f"Qdrant store path: {get_qdrant_store_path()}")


import os

# Get the path to the enriched Indian startup dataset

def get_data_path() -> str:
    base_dir = get_base_dir()
    return os.path.join(get_base_dir(), "Data", "Enriched_Indian_Startup_Dataset.csv")
print(f"Data path: {get_data_path()}")




# Get the Schema Path for Qdrant Store 

def get_schema_path() -> str:
    " GET Qdrant Schema Path "

    base_dir = get_base_dir()

    return os.path.join(base_dir, "schema", "payload_schema.json")
print(f"Schema path: {get_schema_path()}")