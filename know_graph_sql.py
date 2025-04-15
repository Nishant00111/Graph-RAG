import streamlit as st
from langchain_groq import ChatGroq
import json
import os
import base64
import mysql.connector
from sqlalchemy import create_engine
import pandas as pd
import networkx as nx
from pyvis.network import Network

# MySQL connection details
MYSQL_HOST = "localhost"
port = 3306  
MYSQL_USER = "root"
MYSQL_PASSWORD = "Nishant@123"
MYSQL_DATABASE = "test_db"

# Groq API details
GROQ_API_KEY = "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"

# Initialize MySQL connection
def get_mysql_connection():
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        return connection
    except mysql.connector.Error as e:
        st.error(f"Error connecting to MySQL database: {e}")
        return None

# SQLAlchemy engine for pandas
def get_sql_engine():
    try:
        import urllib.parse
        encoded_password = urllib.parse.quote_plus(MYSQL_PASSWORD)
        connection_url = f"mysql+mysqlconnector://{MYSQL_USER}:{encoded_password}@{MYSQL_HOST}:{port}/{MYSQL_DATABASE}"
        engine = create_engine(connection_url)
        return engine
    except Exception as e:
        st.error(f"Error creating SQLAlchemy engine: {e}")
        return None

# Function to fetch database schema
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_db_schema():
    connection = get_mysql_connection()
    if not connection:
        return {}
    
    schema_info = {}
    
    try:
        cursor = connection.cursor()
        
        # Get all tables
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        
        for table in tables:
            # Get columns and their data types
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()
            
            table_info = {}
            for column in columns:
                col_name = column[0]
                col_type = column[1]
                is_key = "PRI" if column[3] == "PRI" else ""
                table_info[col_name] = {"type": col_type, "key": is_key}
            
            schema_info[table] = table_info
            
        # Get foreign keys
        for table in tables:
            cursor.execute(f"""
                SELECT 
                    TABLE_NAME, COLUMN_NAME, 
                    REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM
                    INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE
                    REFERENCED_TABLE_SCHEMA = '{MYSQL_DATABASE}'
                    AND TABLE_NAME = '{table}'
            """)
            
            foreign_keys = cursor.fetchall()
            for fk in foreign_keys:
                table_name, column_name, ref_table, ref_column = fk
                if "foreign_keys" not in schema_info[table_name]:
                    schema_info[table_name]["foreign_keys"] = []
                
                schema_info[table_name]["foreign_keys"].append({
                    "column": column_name,
                    "references": f"{ref_table}({ref_column})"
                })
    
    except mysql.connector.Error as e:
        st.error(f"Error fetching database schema: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
    
    return schema_info

# Function to convert natural language to SQL
def nl_to_sql(query, llm, schema_info):
    # Create a string representation of the schema for the LLM
    schema_str = ""
    for table, columns in schema_info.items():
        schema_str += f"Table: {table}\n"
        schema_str += "Columns:\n"
        
        for col_name, col_info in columns.items():
            if col_name != "foreign_keys":
                key_info = " (Primary Key)" if col_info.get("key") == "PRI" else ""
                schema_str += f"  - {col_name}: {col_info['type']}{key_info}\n"
        
        if "foreign_keys" in columns:
            schema_str += "Foreign Keys:\n"
            for fk in columns["foreign_keys"]:
                schema_str += f"  - {fk['column']} references {fk['references']}\n"
        
        schema_str += "\n"
    
    prompt = f"""Given the following database schema:

{schema_str}

Convert this natural language query to SQL:
"{query}"

Rules:
1. Return ONLY the SQL query without any explanation
2. Use standard SQL syntax compatible with MySQL
3. If the query is not clear, make a reasonable assumption
4. Do not include ```sql or ``` markers
5. Format the SQL query properly with correct indentation
6. When appropriate, limit results to a reasonable number (e.g., TOP 100 or LIMIT 100)
7. Only use tables and columns that exist in the provided schema
8. If you cannot generate a valid SQL query from the input, respond with "Unable to convert to SQL."
"""

    try:
        response = llm.invoke(prompt).content.strip()
        if response.startswith("```sql") and response.endswith("```"):
            response = response[6:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
        
        if response.lower().startswith("unable to convert"):
            return None
            
        return response
    except Exception as e:
        st.error(f"Error generating SQL: {e}")
        return None

# Function to execute SQL and return results
def execute_sql_query(sql_query):
    engine = get_sql_engine()
    if not engine:
        return None, "Failed to connect to database"
    
    try:
        # Execute query and convert to pandas DataFrame
        df = pd.read_sql(sql_query, engine)
        return df, None
    except Exception as e:
        error_message = str(e)
        return None, error_message

# Function to explain SQL results in plain language
def explain_sql_results(sql_query, results_df, user_query, llm):
    if results_df is None or results_df.empty:
        return "No results found for your query."

    results_sample = results_df.head(5).to_string()
    total_rows = len(results_df)
    columns = ", ".join(results_df.columns.tolist())
    
    prompt = f"""Given this user question: "{user_query}"

I ran this SQL query:
{sql_query}

Which returned {total_rows} rows with columns: {columns}

Here's a sample of the results:
{results_sample}

Please provide a concise natural language explanation of these results that answers the original user question. Format your response as a clear, helpful answer without mentioning the SQL query itself.
"""

    try:
        explanation = llm.invoke(prompt).content.strip()
        return explanation
    except Exception as e:
        st.error(f"Error generating explanation: {e}")
        return f"Query returned {total_rows} rows with columns: {columns}"

# Simple database schema visualization function
def create_simple_db_schema_graph(schema_info):
    # Create a network with basic styling
    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # First add all tables as nodes
    for table_name in schema_info:
        # Add table node (blue box)
        net.add_node(
            table_name, 
            label=table_name, 
            shape="box",
            color="#3366CC", 
            size=30,
            title=f"Table: {table_name}"
        )
    
    # Then add columns as child nodes
    for table_name in schema_info:
        for col_name, col_info in schema_info[table_name].items():
            if col_name != "foreign_keys":
                node_id = f"{table_name}_{col_name}"
                
                # Primary keys (gold color)
                if col_info.get("key") == "PRI":
                    node_color = "#FFD700"
                    node_shape = "diamond"
                    node_label = f"{col_name} (PK)"
                    node_size = 20
                else:
                    # Regular columns (light gray)
                    node_color = "#CCCCCC"
                    node_shape = "dot"
                    node_label = col_name
                    node_size = 15
                
                net.add_node(
                    node_id,
                    label=node_label,
                    shape=node_shape,
                    color=node_color,
                    size=node_size,
                    title=f"Column: {col_name}\nType: {col_info['type']}"
                )
                
                # Connect table to its column
                net.add_edge(table_name, node_id, color="#888888", width=0.5)
    
    # Collect all table-to-table relationships before adding them
    table_relationships = []
    
    # Add FK-PK connections and collect table relationships
    for table_name, table_info in schema_info.items():
        if "foreign_keys" in table_info:
            for fk in table_info["foreign_keys"]:
                source_col = fk['column']
                source_node = f"{table_name}_{source_col}"
                
                # Mark columns that are foreign keys with orange color
                for node in net.nodes:
                    if node['id'] == source_node:
                        node['color'] = "#FF9900"
                        node['label'] = f"{source_col} (FK)"
                        node['title'] = f"Foreign Key: {source_col}"
                        break
                
                # Parse referenced table and column
                ref_parts = fk['references'].split('(')
                ref_table = ref_parts[0]
                ref_column = ref_parts[1].replace(')', '')
                ref_node = f"{ref_table}_{ref_column}"
                
                # Add relationship between foreign key and primary key
                net.add_edge(
                    source_node, 
                    ref_node,
                    color="red",
                    arrows="to",
                    title=f"References",
                    width=1.5,
                    dashes=True
                )
                
                # Collect table relationship
                table_relationships.append({
                    'from': table_name,
                    'to': ref_table,
                    'label': f"{source_col} -> {ref_column}"
                })
    
    # Add table-to-table relationships (with thick, highly visible edges)
    for rel in table_relationships:
        net.add_edge(
            rel['from'],
            rel['to'],
            color="#FF5733",  # Bright orange-red
            arrows="to",
            title=rel['label'],
            width=5,         # Make lines thicker
            arrowStrikethrough=False,
            smooth={'enabled': True, 'type': 'curvedCW', 'roundness': 0.2}
        )
    
    # Add improved controls and styling
    options = {
        "nodes": {
            "font": {"size": 14},
            "scaling": {"min": 10, "max": 30}
        },
        "edges": {
            "color": {"inherit": False},
            "smooth": {
                "enabled": True,
                "type": "continuous"
            },
            "hoverWidth": 2
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -60,
                "centralGravity": 0.01,
                "springLength": 150,  # Increase space between nodes
                "springConstant": 0.05
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.3,
            "stabilization": {
                "enabled": True,
                "iterations": 1000,
                "updateInterval": 25
            }
        },
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "keyboard": True,
            "tooltipDelay": 200
        }
    }
    
    net.set_options(json.dumps(options))
    
    # Save to HTML file
    html_file = "database_schema.html"
    net.save_graph(html_file)
    
    # Add enhanced legend and full-screen button
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Add improved legend and controls
    custom_html = """
    <style>
        #mynetwork {
            width: 100%;
            height: 700px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        #legend {
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            z-index: 1000;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .legend-item {
            margin: 5px 0;
            display: flex;
            align-items: center;
        }
        .color-box {
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 5px;
            border-radius: 3px;
        }
        .line-box {
            display: inline-block;
            width: 20px;
            height: 2px;
            margin-right: 5px;
        }
        .table-rel-line {
            display: inline-block;
            width: 20px;
            height: 5px;  /* Thicker line */
            margin-right: 5px;
            background: #FF5733;
        }
        .fullscreen-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 1000;
            padding: 5px 10px;
            background: #3366CC;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
        }
        .controls {
            position: absolute;
            bottom: 10px;
            left: 10px;
            z-index: 1000;
            background: white;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .btn {
            margin: 2px;
            padding: 3px 8px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 3px;
            cursor: pointer;
        }
        .btn:hover {
            background: #e0e0e0;
        }
    </style>
    
    <div id="legend">
        <div style="font-weight: bold; margin-bottom: 8px;">Legend:</div>
        <div class="legend-item"><div class="color-box" style="background:#3366CC"></div> Table</div>
        <div class="legend-item"><div class="color-box" style="background:#FFD700"></div> Primary Key</div>
        <div class="legend-item"><div class="color-box" style="background:#FF9900"></div> Foreign Key</div>
        <div class="legend-item"><div class="color-box" style="background:#CCCCCC"></div> Regular Column</div>
        <div class="legend-item"><div class="line-box" style="background:red"></div> FK-PK Connection</div>
        <div class="legend-item"><div class="table-rel-line"></div> Table Relationship</div>
    </div>
    
    <button class="fullscreen-btn" onclick="toggleFullscreen()">Fullscreen</button>
    
    <div class="controls">
        <button class="btn" onclick="network.fit()">Fit View</button>
        <button class="btn" onclick="network.stabilize()">Stabilize</button>
        <button class="btn" onclick="togglePhysics()">Toggle Physics</button>
        <button class="btn" onclick="toggleTableConnections()">Toggle Table Connections</button>
    </div>
    
    <script>
        var showTableConnections = true;
        
        function toggleFullscreen() {
            var elem = document.getElementById('mynetwork');
            if (!document.fullscreenElement) {
                elem.requestFullscreen().catch(err => {
                    console.log("Error attempting fullscreen:", err);
                });
            } else {
                document.exitFullscreen();
            }
        }
        
        function togglePhysics() {
            var options = network.physics.options;
            options.enabled = !options.enabled;
            network.setOptions({ physics: options });
        }
        
        // Function to toggle visibility of table connections
        function toggleTableConnections() {
            showTableConnections = !showTableConnections;
            
            // Get all edges
            var edges = network.body.data.edges;
            var edgeIds = edges.getIds();
            
            for (var i = 0; i < edgeIds.length; i++) {
                var edge = edges.get(edgeIds[i]);
                
                // Check if this is a table-to-table connection
                // (assuming table nodes don't have underscores in their IDs)
                if (!edge.from.includes('_') && !edge.to.includes('_')) {
                    edges.update({
                        id: edgeIds[i],
                        hidden: !showTableConnections
                    });
                }
            }
        }
    </script>
    """
    
    html_content = html_content.replace('</body>', custom_html + '</body>')
    
    if os.path.exists(html_file):
        os.remove(html_file)
    
    return html_content

# Streamlit app
def main():
    st.set_page_config(
        page_title="Simple Database Query & Visualization",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Database Query & Visualization Tool")
    
    # Initialize LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-70b-8192",
        temperature=0.1
    )
    
    # Tab selection
    tab1, tab2 = st.tabs(["Ask Questions (Text-to-SQL)", "View Database Schema"])
    
    # Load schema info
    with st.spinner("Loading database schema..."):
        schema_info = get_db_schema()
    
    # Text-to-SQL Tab
    with tab1:
        st.write("Ask questions about your database in plain English")
        
        # Test connection button
        if st.button("Test Database Connection"):
            connection = get_mysql_connection()
            if connection:
                st.success("Successfully connected to database!")
                connection.close()
        
        # Show schema in expander
        with st.expander("View Database Tables"):
            if schema_info:
                for table, columns in schema_info.items():
                    st.subheader(f"Table: {table}")
                    
                    # Show columns
                    col_data = []
                    for col_name, col_info in columns.items():
                        if col_name != "foreign_keys":
                            key_info = "Primary Key" if col_info.get("key") == "PRI" else ""
                            col_data.append([col_name, col_info['type'], key_info])
                    
                    st.table(pd.DataFrame(col_data, columns=["Column", "Type", "Key"]))
                    
                    # Show foreign keys
                    if "foreign_keys" in columns:
                        st.write("**Foreign Keys:**")
                        for fk in columns["foreign_keys"]:
                            st.write(f"- {fk['column']} connects to {fk['references']}")
                    st.write("---")
            else:
                st.warning("Could not load database schema. Check connection.")
        
        # Natural language query input
        nl_query = st.text_area("Type your question here:", 
                              placeholder="Example: Show me all customers who placed orders in the last month")
        
        if nl_query and schema_info:
            with st.spinner("Converting to SQL..."):
                sql_query = nl_to_sql(nl_query, llm, schema_info)
                
            if sql_query:
                st.subheader("Generated SQL Query:")
                st.code(sql_query, language="sql")
                
                # Execute query button
                if st.button("Run Query"):
                    with st.spinner("Running query..."):
                        results_df, error = execute_sql_query(sql_query)
                    
                    if error:
                        st.error(f"Error: {error}")
                    elif results_df is not None:
                        # Display results
                        st.subheader("Results")
                        st.dataframe(results_df)
                        
                        # Show explanation
                        with st.spinner("Generating explanation..."):
                            explanation = explain_sql_results(sql_query, results_df, nl_query, llm)
                        
                        st.subheader("What This Means:")
                        st.write(explanation)
                        
                        # Download button
                        if not results_df.empty:
                            csv = results_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="query_results.csv">Download Results (CSV)</a>'
                            st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("Could not convert your question to SQL. Please try rephrasing.")
    
    # Database Schema Graph Tab
    with tab2:
        st.write("See how your tables are connected")
        
        if schema_info:
            with st.spinner("Creating database diagram..."):
                html_content = create_simple_db_schema_graph(schema_info)
                
                # Display help text
                st.markdown("""
                ### How to use the diagram:
                - **Blue boxes** are tables
                - **Gold diamonds** are primary keys
                - **Orange circles** are foreign keys
                - **Red lines** show connections between tables
                - Drag nodes to move them around
                - Scroll to zoom in/out
                """)
                
                # Display the graph
                st.components.v1.html(html_content, height=700)
        else:
            st.warning("Could not load database schema. Check connection.")

    # Sidebar
    st.sidebar.title("Connection Info")
    st.sidebar.info(f"""
    **Current Connection:**
    - Host: {MYSQL_HOST}
    - Database: {MYSQL_DATABASE}
    - User: {MYSQL_USER}
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")