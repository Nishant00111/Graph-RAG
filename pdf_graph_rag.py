# Install required packages
# pip install streamlit langchain langchain-experimental neo4j PyPDF2 sentence-transformers langchain-groq

import streamlit as st
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import json
import PyPDF2
import re
import numpy as np
import os
import base64

# Neo4j Aura connection details
NEO4J_URI = "bolt+ssc://544ab938.databases.neo4j.io"  # Use bolt+ssc for SSL connection
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "WDV_Rpvja3HTSM7eZ6ctS0MU5nschz5RXGn8XI9ZVCM"

# Groq API details
GROQ_API_KEY = "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"

# Initialize Neo4j driver and embedding model
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,-]', '', text)
    return text

# Function to read and chunk large PDF content
def read_pdf(file, chunk_size=5000):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        full_text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += f" {clean_text(extracted)}"
        
        if not full_text:
            st.warning("No text extracted from PDF.")
            return []
        
        chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
        return chunks
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return []

# Function to enhance nodes with descriptions from PDF context
def enhance_nodes_with_context(nodes, text_chunks):
    node_descriptions = {node.id: node.id for node in nodes}
    for chunk in text_chunks:
        for node in nodes:
            if node.id.lower() in chunk.lower():
                start_idx = max(0, chunk.lower().index(node.id.lower()) - 50)
                end_idx = min(len(chunk), chunk.lower().index(node.id.lower()) + len(node.id) + 50)
                context = chunk[start_idx:end_idx].strip()
                node_descriptions[node.id] = context
    return node_descriptions

# Function to insert graph data with embeddings into Neo4j Aura
def insert_graph_data(tx, nodes, relationships, node_descriptions):
    node_texts = [f"{node.id} ({node.type}): {node_descriptions[node.id]}" for node in nodes]
    embeddings = embedder.encode(node_texts, convert_to_numpy=True).tolist()
    
    for node, embedding in zip(nodes, embeddings):
        tx.run("""
            MERGE (n:Node {id: $id, type: $type})
            SET n.embedding = $embedding, n.description = $description
        """, id=node.id, type=node.type, embedding=embedding, description=node_descriptions[node.id])
    
    for rel in relationships:
        tx.run("""
            MATCH (source:Node {id: $source_id})
            MATCH (target:Node {id: $target_id})
            MERGE (source)-[:RELATIONSHIP {type: $rel_type}]->(target)
        """, source_id=rel.source.id, target_id=rel.target.id, rel_type=rel.type)

# Function to query Neo4j using hybrid vector + keyword search
def query_neo4j(query, llm):
    with driver.session() as session:
        query_lower = query.lower()
        query_words = [word for word in query_lower.split() if len(word) > 2]
        
        query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()[0]
        
        cypher_query = """
        CALL db.index.vector.queryNodes('nodeEmbeddingIndex', 5, $query_embedding)
        YIELD node, score
        MATCH (node)-[r:RELATIONSHIP]->(m:Node)
        WHERE score > 0.6 OR any(word IN $query_words WHERE toLower(node.id) CONTAINS word OR toLower(m.id) CONTAINS word)
        RETURN node.id AS source, r.type AS rel_type, m.id AS target, node.description AS source_desc, m.description AS target_desc, score
        ORDER BY score DESC
        LIMIT 10
        """
        result = session.run(cypher_query, query_embedding=query_embedding, query_words=query_words)
        
        subgraph = []
        for record in result:
            source = record["source"]
            rel_type = record["rel_type"].lower()
            target = record["target"]
            source_desc = record["source_desc"] or source
            target_desc = record["target_desc"] or target
            subgraph.append(f"{source} {rel_type} {target} (Context: {source_desc} -> {target_desc})")
        
        if not subgraph:
            st.warning(f"No subgraph found for query: {query}")
            prompt = f"Answer the query '{query}' in one sentence starting with 'In simple terms,' based on general knowledge."
            return llm.invoke(prompt).content.strip()
        
        subgraph_text = ". ".join(subgraph) + "."
        print("Raw subgraph retrieved:", subgraph_text)
        
        prompt = f"Using the following information from a knowledge graph: '{subgraph_text}', answer the query '{query}' in one sentence starting with 'In simple terms,'. If the graph lacks details, supplement with general knowledge."
        try:
            refined_answer = llm.invoke(prompt).content
            return refined_answer.strip()
        except Exception as e:
            st.error(f"Error refining answer with LLM: {str(e)}")
            return subgraph_text

# Function to generate and return HTML visualization as string
def generate_html_visualization(graph_document):
    nodes = []
    edges = []
    
    for i, node in enumerate(graph_document.nodes):
        nodes.append({"id": i, "label": f"{node.id} ({node.type})"})
    
    node_id_map = {node.id: i for i, node in enumerate(graph_document.nodes)}
    for rel in graph_document.relationships:
        try:
            source_id = node_id_map[rel.source.id]
            target_id = node_id_map[rel.target.id]
            edges.append({"from": source_id, "to": target_id, "label": rel.type, "arrows": "to"})
        except KeyError as e:
            st.warning(f"Skipping relationship due to missing node: {e}")
            continue
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Knowledge Graph Visualization</title>
        <script src="https://visjs.github.io/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            #network {{ width: 100%; height: 600px; border: 1px solid lightgray; }}
        </style>
    </head>
    <body>
        <div id="network"></div>
        <script>
            var nodes = new vis.DataSet({json.dumps(nodes, indent=2)});
            var edges = new vis.DataSet({json.dumps(edges, indent=2)});
            var container = document.getElementById('network');
            var data = {{ nodes: nodes, edges: edges }};
            var options = {{
                nodes: {{ shape: 'dot', size: 20, font: {{ size: 14 }} }},
                edges: {{ font: {{ size: 12 }}, arrows: {{ to: {{ enabled: true, scaleFactor: 1 }} }} }},
                physics: {{ enabled: true }},
                interaction: {{ dragNodes: true, zoomView: true, dragView: true }}
            }};
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """
    return html_content

# Function to process PDF and build graph
def build_graph(pdf_file, llm):
    llm_transformer = LLMGraphTransformer(llm=llm)
    text_chunks = read_pdf(pdf_file)
    
    if not text_chunks:
        return None
    
    st.write("Extracted PDF Text Preview (first 500 characters of first chunk):")
    st.write(text_chunks[0][:500])
    
    all_nodes = []
    all_relationships = []
    for chunk in text_chunks:
        documents = [Document(page_content=chunk)]
        try:
            graph_documents = llm_transformer.convert_to_graph_documents(documents)
            if graph_documents and graph_documents[0].nodes:
                all_nodes.extend(graph_documents[0].nodes)
                all_relationships.extend(graph_documents[0].relationships)
        except Exception as e:
            st.error(f"Error processing chunk: {str(e)}")
            continue
    
    all_nodes_dict = {(node.id, node.type): node for node in all_nodes}
    for rel in all_relationships:
        all_nodes_dict[(rel.source.id, rel.source.type)] = rel.source
        all_nodes_dict[(rel.target.id, rel.target.type)] = rel.target
    unique_nodes = list(all_nodes_dict.values())
    
    unique_relationships = list({(rel.source.id, rel.target.id, rel.type): rel for rel in all_relationships}.values())
    
    if not unique_nodes:
        st.warning("No nodes generated from the PDF.")
        return None
    
    node_descriptions = enhance_nodes_with_context(unique_nodes, text_chunks)
    
    source_doc = Document(page_content="Aggregated from uploaded PDF")
    aggregated_graph = GraphDocument(nodes=unique_nodes, relationships=unique_relationships, source=source_doc)
    
    with driver.session() as session:
        session.execute_write(insert_graph_data, aggregated_graph.nodes, aggregated_graph.relationships, node_descriptions)
    st.success("Graph data with embeddings successfully inserted into Neo4j Aura.")
    
    with driver.session() as session:
        result = session.run("SHOW INDEXES WHERE name = 'nodeEmbeddingIndex'")
        index_exists = any(record for record in result)
        if not index_exists:
            session.run("""
                CALL db.index.vector.createNodeIndex(
                    'nodeEmbeddingIndex',
                    'Node',
                    'embedding',
                    384,
                    'cosine'
                )
            """)
            st.success("Vector index 'nodeEmbeddingIndex' created.")
        else:
            st.info("Vector index 'nodeEmbeddingIndex' already exists, skipping creation.")
    
    return aggregated_graph

# Streamlit app
def main():
    st.title("Knowledge Graph RAG App")
    st.write("Upload a PDF to build a knowledge graph and query it.")

    # Initialize LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-70b-8192",  # Switch to "llama3-8b-8192" for speed if needed
        temperature=0.1,
        max_retries=2
    )

    # PDF upload
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF and building graph..."):
            aggregated_graph = build_graph(uploaded_file, llm)
        
        if aggregated_graph:
            # Display graph visualization
            html_content = generate_html_visualization(aggregated_graph)
            st.components.v1.html(html_content, height=600)
            
            # Query interface
            st.subheader("Query the Knowledge Graph")
            query = st.text_input("Enter your query:")
            if query:
                with st.spinner("Fetching answer..."):
                    answer = query_neo4j(query, llm)
                st.write("**Answer:**", answer)

    # Cleanup
    if st.button("Close Connection"):
        driver.close()
        st.success("Neo4j connection closed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        if 'driver' in globals():
            driver.close()