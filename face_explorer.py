import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from qdrant_client import QdrantClient
import umap
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import base64
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Face Vector Explorer",
    page_icon="👤",
    layout="wide"
)

# Initialize Qdrant client
@st.cache_resource
def get_client():
    return QdrantClient(url="http://localhost:6333")

client = get_client()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .face-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .similar-faces {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 20px;
    }
    .face-img {
        border-radius: 5px;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .face-img:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">👤 Face Vector Database Explorer</h1>', unsafe_allow_html=True)

# Sidebar filters
with st.sidebar:
    st.header("🔍 Navigation & Filters")
    
    # Collection stats - FIXED API
    try:
        collection_info = client.get_collection("faces")
        
        # Get vectors count - different ways depending on Qdrant version
        vectors_count = 0
        if hasattr(collection_info, 'vectors_count'):
            vectors_count = collection_info.vectors_count
        elif hasattr(collection_info, 'points_count'):
            vectors_count = collection_info.points_count
        elif hasattr(collection_info, 'config'):
            # Try to get from config
            vectors_count = collection_info.config.params.vectors.size if hasattr(collection_info.config.params, 'vectors') else 0
        
        # Get vector dimensions
        vector_dim = 0
        if hasattr(collection_info.config.params, 'vectors'):
            if hasattr(collection_info.config.params.vectors, 'size'):
                vector_dim = collection_info.config.params.vectors.size
            elif isinstance(collection_info.config.params.vectors, dict):
                # Might be a dict in newer versions
                vector_dim = collection_info.config.params.vectors.get('size', 0)
        
        st.metric("Total Faces", f"{vectors_count:,}")
        st.metric("Vector Dimensions", vector_dim)
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.warning("'faces' collection not found or connection failed!")
        
        # Try to list all collections
        try:
            collections = client.get_collections()
            st.info(f"Available collections: {[c.name for c in collections.collections]}")
        except:
            pass
        
        st.stop()
    
    # Data sampling
    sample_size = st.slider("Sample size for visualization", 100, 5000, 1000, 100)
    
    # Visualization settings
    st.subheader("Visualization Settings")
    viz_method = st.selectbox("Reduction method", ["UMAP", "t-SNE", "PCA"])
    color_by = st.selectbox("Color by", ["Clusters", "Similarity", "Metadata"])
    
    # Search
    st.subheader("🔎 Search")
    search_query = st.text_input("Search in metadata")
    
    # Filter by payload
    st.subheader("Filters")
    filter_age = st.slider("Age range", 0, 100, (0, 100))
    filter_gender = st.multiselect("Gender", ["male", "female", "other"])
    
    # Actions
    if st.button("🔄 Refresh Data"):
        st.rerun()

# Get face data (moved outside tabs for shared access)
@st.cache_data(ttl=300)
def get_face_data(limit):
    try:
        points, _ = client.scroll(
            collection_name="faces",
            limit=limit,
            with_payload=True,
            with_vectors=True
        )
        return points
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []

# Load data once
points = get_face_data(sample_size)

# Get collection info for later use
try:
    collection_info = client.get_collection("faces")
except:
    collection_info = None

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Search & Explore", "👥 Similarity Search", "📁 Collection Info"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Face Vector Space Visualization")
        
        if points and len(points) > 0:
            vectors = np.array([p.vector for p in points])
            
            # Dimensionality reduction
            try:
                if viz_method == "UMAP":
                    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(points)-1))
                    embeddings_2d = reducer.fit_transform(vectors)
                    
                elif viz_method == "t-SNE":
                    from sklearn.manifold import TSNE
                    reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(points)-1))
                    embeddings_2d = reducer.fit_transform(vectors)
                    
                elif viz_method == "PCA":
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=2, random_state=42)
                    embeddings_2d = reducer.fit_transform(vectors)
                
                # Prepare dataframe
                df = pd.DataFrame({
                    'x': embeddings_2d[:, 0],
                    'y': embeddings_2d[:, 1],
                    'id': [str(p.id) for p in points],
                    'vector_norm': [np.linalg.norm(v) for v in vectors]
                })
                
                # Add payload data
                payload_fields = []
                if points[0].payload:
                    payload_fields = list(points[0].payload.keys())
                    for field in payload_fields:
                        df[field] = [p.payload.get(field, '') for p in points]
                
                # Create interactive plot
                if color_by == "Similarity":
                    color_column = 'vector_norm'
                elif color_by == "Metadata" and payload_fields:
                    color_column = payload_fields[0]
                else:
                    color_column = 'id'
                
                fig = px.scatter(
                    df, x='x', y='y',
                    color=color_column,
                    hover_data=df.columns.tolist(),
                    title=f"{len(points)} Faces in 2D Space ({viz_method})",
                    size_max=10
                )
                
                fig.update_traces(
                    marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')),
                    selector=dict(mode='markers')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Visualization error: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.info("Try reducing sample size or using UMAP")
        
        else:
            st.warning("No data found in 'faces' collection")
        
    with col2:
        st.subheader("Collection Statistics")
        
        if points and len(points) > 0:
            vectors = np.array([p.vector for p in points])
            
            # Basic stats
            stats_data = {
                'Metric': ['Sample Size', 'Vector Dimensions', 'Average Vector Norm', 'Min Norm', 'Max Norm'],
                'Value': [
                    len(points),
                    vectors.shape[1] if len(points) > 0 else 0,
                    f"{np.mean([np.linalg.norm(v) for v in vectors]):.4f}" if len(points) > 0 else 0,
                    f"{np.min([np.linalg.norm(v) for v in vectors]):.4f}" if len(points) > 0 else 0,
                    f"{np.max([np.linalg.norm(v) for v in vectors]):.4f}" if len(points) > 0 else 0
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, hide_index=True)
            
            # Payload field distribution
            if points[0].payload:
                payload_fields = list(points[0].payload.keys())
                if payload_fields:
                    st.subheader("Payload Distribution")
                    selected_field = st.selectbox("Select field", payload_fields, key="dist_field")
                    if selected_field:
                        values = [str(p.payload.get(selected_field, '')) for p in points]
                        value_counts = pd.Series(values).value_counts()
                        
                        if len(value_counts) > 20:
                            st.warning(f"Too many unique values ({len(value_counts)}). Showing top 20.")
                            value_counts = value_counts.head(20)
                        
                        fig_pie = px.pie(
                            names=value_counts.index, 
                            values=value_counts.values,
                            title=f"Distribution of {selected_field}"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("Browse Individual Faces")
    
    if points and len(points) > 0:
        # Grid view of faces
        cols_per_row = 5
        num_to_display = min(50, len(points))
        
        for i in range(0, num_to_display, cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < num_to_display:
                    point = points[idx]
                    with col:
                        # Create a unique key for each button
                        button_key = f"similar_{point.id}_{idx}"
                        
                        # Display face image if URL exists in payload
                        if point.payload and 'image_url' in point.payload:
                            try:
                                response = requests.get(point.payload['image_url'], timeout=2)
                                img = Image.open(BytesIO(response.content))
                                col.image(img, caption=f"ID: {point.id}", use_column_width=True)
                            except:
                                col.markdown(f"**Face {point.id}**")
                        
                        # Show metadata in expander
                        with st.expander(f"Details {point.id}", expanded=False):
                            st.write("**ID:**", point.id)
                            if point.payload:
                                for key, value in point.payload.items():
                                    if key != 'image_url':
                                        st.write(f"**{key}:**", value)
                            if point.vector:
                                st.write("**Vector (first 5):**", point.vector[:5])
                            
                            # Find similar faces button - stores in session state
                            if st.button(f"Find Similar", key=button_key):
                                st.session_state['search_vector'] = point.vector
                                st.session_state['search_id'] = point.id
                                st.success(f"Selected face {point.id} for similarity search! Go to Similarity Search tab.")
        
        st.info(f"Showing {num_to_display} of {len(points)} faces. Use sidebar to adjust sample size.")
    else:
        st.warning("No faces to display")

with tab3:
    st.subheader("Similarity Search")
    
    # Initialize session state for search
    if 'search_vector' not in st.session_state:
        st.session_state['search_vector'] = None
    if 'search_id' not in st.session_state:
        st.session_state['search_id'] = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Search by vector or example
        search_type = st.radio("Search by:", ["Example Face", "Custom Vector"])
        
        query_vector = None
        
        if search_type == "Example Face":
            if points and len(points) > 0:
                example_options = [str(p.id) for p in points[:100]]
                example_options.insert(0, "Select a face")
                example_id = st.selectbox("Select example face", example_options)
                
                # Check if we have a stored search vector from tab2
                if st.session_state['search_vector'] is not None:
                    st.info(f"Pre-selected face ID: {st.session_state['search_id']}")
                    if st.button("Use Pre-selected Face"):
                        query_vector = st.session_state['search_vector']
                        st.success(f"Using face {st.session_state['search_id']}")
                
                if example_id and example_id != "Select a face":
                    point = next((p for p in points if str(p.id) == example_id), None)
                    if point and st.button("Search with Selected Face"):
                        query_vector = point.vector
                        st.session_state['last_search_vector'] = query_vector
                        st.session_state['last_search_id'] = example_id
        
        else:  # Custom vector
            # Default vector from first point if available
            default_vector = ""
            if points and len(points) > 0 and points[0].vector:
                default_vector = ", ".join([str(x) for x in points[0].vector[:10]]) + ", ..."
            
            vector_input = st.text_area("Paste vector (comma-separated):", default_vector, height=150)
            
            if st.button("Search with Custom Vector"):
                try:
                    query_vector = [float(x.strip()) for x in vector_input.split(',')]
                    st.success(f"Vector loaded with {len(query_vector)} dimensions")
                except:
                    st.error("Invalid vector format. Please enter comma-separated numbers.")
        
        # Search parameters
        top_k = st.slider("Number of results", 5, 100, 20)
        score_threshold = st.slider("Minimum similarity score", 0.0, 1.0, 0.5, 0.05)
    
    with col2:
        # Perform search if we have a query vector
        current_query = query_vector or st.session_state.get('search_vector')
        
        if current_query is not None:
            try:
                results = client.search(
                    collection_name="faces",
                    query_vector=current_query,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False
                )
                
                if results:
                    st.subheader(f"Top {len(results)} Similar Faces")
                    
                    # Create results dataframe
                    results_data = []
                    for i, result in enumerate(results):
                        row = {
                            'Rank': i+1,
                            'ID': result.id,
                            'Similarity Score': f"{result.score:.4f}",
                            'Distance': f"{1 - result.score:.4f}"  # Convert similarity to distance
                        }
                        if result.payload:
                            for key, value in result.payload.items():
                                if isinstance(value, (str, int, float, bool)):
                                    row[key] = value
                        results_data.append(row)
                    
                    df_results = pd.DataFrame(results_data)
                    st.dataframe(df_results, use_container_width=True, height=400)
                    
                    # Display individual results with images
                    for i, result in enumerate(results[:10]):  # Limit to 10 for display
                        with st.expander(f"#{i+1}: ID {result.id} (Score: {result.score:.3f})"):
                            if result.payload:
                                for key, value in result.payload.items():
                                    if key != 'image_url':
                                        st.write(f"**{key}:** {value}")
                                
                                # Display image if available
                                if 'image_url' in result.payload:
                                    try:
                                        response = requests.get(result.payload['image_url'], timeout=3)
                                        img = Image.open(BytesIO(response.content))
                                        st.image(img, caption=f"Face {result.id}", width=200)
                                    except Exception as img_error:
                                        st.write(f"Image unavailable: {img_error}")
                            else:
                                st.write("No payload available")
                else:
                    st.warning("No results found. Try lowering the score threshold.")
                    
            except Exception as e:
                st.error(f"Search error: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.info("Select a face or enter a vector to perform similarity search")
            
            # Show last search if available
            if 'last_search_id' in st.session_state:
                st.info(f"Last search was for face ID: {st.session_state['last_search_id']}")

with tab4:
    st.subheader("Collection Details")
    
    if collection_info:
        try:
            # Collection info in expandable sections
            with st.expander("📊 Collection Configuration", expanded=True):
                # Convert to dict for display
                if hasattr(collection_info.config, 'dict'):
                    config_dict = collection_info.config.dict()
                else:
                    # Try to convert manually
                    config_dict = {}
                    if hasattr(collection_info.config, 'params'):
                        params = collection_info.config.params
                        if hasattr(params, 'vectors'):
                            vectors_info = params.vectors
                            if hasattr(vectors_info, 'size'):
                                config_dict['vector_size'] = vectors_info.size
                            elif isinstance(vectors_info, dict):
                                config_dict.update(vectors_info)
                
                st.json(config_dict)
            
            with st.expander("📈 Collection Status", expanded=False):
                status_dict = {}
                
                # Try different attribute names
                for attr in ['vectors_count', 'points_count', 'segments_count', 'status']:
                    if hasattr(collection_info, attr):
                        status_dict[attr] = getattr(collection_info, attr)
                
                if hasattr(collection_info, 'optimizer_status'):
                    status_dict['optimizer_status'] = collection_info.optimizer_status.status
                
                st.json(status_dict)
            
            with st.expander("🔧 Payload Schema", expanded=False):
                # Try to infer schema from sample data
                if points and len(points) > 0 and points[0].payload:
                    sample_payload = points[0].payload
                    schema = {}
                    for key, value in sample_payload.items():
                        schema[key] = type(value).__name__
                    st.json(schema)
                else:
                    st.info("No payload schema available")
            
            # Export options
            st.subheader("📤 Data Export")
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                export_format = st.selectbox("Export format", ["CSV", "JSON"])
                include_vectors = st.checkbox("Include vectors", value=False)
                include_payload = st.checkbox("Include payload", value=True)
            
            with col_export2:
                if st.button("📥 Export Data", type="primary"):
                    with st.spinner("Preparing export..."):
                        # Prepare data for export
                        export_data = []
                        for p in points:
                            item = {'id': p.id}
                            if include_payload and p.payload:
                                item.update(p.payload)
                            if include_vectors and p.vector:
                                item['vector'] = p.vector
                            export_data.append(item)
                        
                        df_export = pd.DataFrame(export_data)
                        
                        if export_format == "CSV":
                            csv = df_export.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="faces_export.csv",
                                mime="text/csv"
                            )
                        elif export_format == "JSON":
                            json_str = df_export.to_json(orient='records', indent=2)
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name="faces_export.json",
                                mime="application/json"
                            )
            
            # Quick stats
            st.subheader("📋 Quick Statistics")
            if points and len(points) > 0:
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Sample Size", len(points))
                with col_stat2:
                    if points[0].payload:
                        st.metric("Payload Fields", len(points[0].payload.keys()))
                    else:
                        st.metric("Payload Fields", 0)
                with col_stat3:
                    st.metric("Vector Dims", len(points[0].vector) if points[0].vector else 0)
        
        except Exception as e:
            st.error(f"Error displaying collection info: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning("Collection info not available")

# Footer
st.markdown("---")
st.markdown("**Qdrant Face Database Explorer** • Built with Streamlit")

# Debug info (optional - can be removed)
with st.sidebar.expander("Debug Info"):
    st.write(f"Points loaded: {len(points) if points else 0}")
    if points and len(points) > 0:
        st.write(f"First point ID: {points[0].id}")
        if points[0].payload:
            st.write(f"Payload keys: {list(points[0].payload.keys())}")
    
    # Show collection info structure
    if collection_info:
        st.write("Collection Info attributes:")
        st.write([attr for attr in dir(collection_info) if not attr.startswith('_')])