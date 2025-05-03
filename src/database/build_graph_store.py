from langchain_community.graphs import Neo4jGraph
import pandas as pd
from tqdm import tqdm

# Neo4j setup (update credentials as needed)
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="your_password"
)

# Load your dataset
df = pd.read_csv("data/netflix_dataset.csv")

# Define and push nodes & relationships dynamically
graph.query("MATCH (n) DETACH DELETE n")  # Clear the graph

for _, row in tqdm(df.iterrows(), total=len(df)):
    title = row.get("title", "Unknown")
    genre = row.get("genre", "Unknown")
    year = row.get("release_year", "Unknown")
    type_ = row.get("type", "Unknown")

    query = f"""
    MERGE (m:Movie {{title: '{title}', type: '{type_}', year: '{year}'}})
    MERGE (g:Genre {{name: '{genre}'}})
    MERGE (m)-[:IN_GENRE]->(g)
    """
    graph.query(query)

print("✅ Graph nodes and relationships pushed to Neo4j.")