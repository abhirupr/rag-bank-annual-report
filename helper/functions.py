import time
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks_langchain import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
import pandas as pd

# --- Databricks and Vector Search Utilities ---
def is_databricks_environment() -> bool:
    """
    Check if the code is running inside a Databricks environment.
    Returns:
        bool: True if running in Databricks, False otherwise.
    """
    import os
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

def is_table_empty(spark, table_name):
    """Return True if the Spark table is empty or does not exist."""
    try:
        count = spark.sql(f"SELECT COUNT(*) as count FROM {table_name}").collect()[0]['count']
        return count == 0
    except Exception as e:
        # If table doesn't exist, treat as empty
        return True

def wait_for_vs_endpoint_to_be_ready(vsc: VectorSearchClient, vs_endpoint_name: str) -> dict:
    """
    Wait until the specified Vector Search endpoint is online.
    Args:
        vsc (VectorSearchClient): The vector search client.
        vs_endpoint_name (str): The name of the endpoint.
    Returns:
        dict: Endpoint details if ready.
    Raises:
        Exception: If endpoint is not ready after timeout.
    """
    for i in range(180):
        try:
            endpoint = vsc.get_endpoint(vs_endpoint_name)
            status = endpoint.get("endpoint_status", {}).get("state", "").upper()
            if status == "ONLINE":
                return endpoint
        except Exception:
            pass
        time.sleep(10)
    raise Exception(f"Endpoint {vs_endpoint_name} not ready after 30 minutes")

def wait_for_index_to_be_ready(vsc: VectorSearchClient, vs_endpoint_name: str, index_name: str) -> dict:
    """
    Wait until the specified index is ready.
    Args:
        vsc (VectorSearchClient): The vector search client.
        vs_endpoint_name (str): The endpoint name.
        index_name (str): The index name.
    Returns:
        dict: Index details if ready.
    Raises:
        Exception: If index is not ready after timeout.
    """
    for i in range(180):
        try:
            idx = vsc.get_index(vs_endpoint_name, index_name).describe()
            status = idx.get("status", {}).get("ready", False)
            if status:
                return idx
        except Exception:
            pass
        time.sleep(10)
    raise Exception(f"Index {index_name} not ready after 30 minutes")

def index_exists(vsc: VectorSearchClient, vs_endpoint_name: str, index_name: str) -> bool:
    """
    Check if a vector search index exists.
    Args:
        vsc (VectorSearchClient): The vector search client.
        vs_endpoint_name (str): The endpoint name.
        index_name (str): The index name.
    Returns:
        bool: True if index exists, False otherwise.
    """
    try:
        vsc.get_index(vs_endpoint_name, index_name).describe()
        return True
    except Exception:
        return False

def create_vector_search_endpoint(vsc: VectorSearchClient, endpoint_name: str) -> dict:
    """
    Create a new vector search endpoint or return the existing one.
    Args:
        vsc (VectorSearchClient): The vector search client.
        endpoint_name (str): The endpoint name.
    Returns:
        dict: Endpoint details.
    Raises:
        Exception: If endpoint creation fails.
    """
    try:
        endpoint = vsc.get_endpoint(endpoint_name)
        print(f"✅ Using existing endpoint: {endpoint_name}")
        return endpoint
    except:
        print(f"📝 Creating endpoint: {endpoint_name}")
        try:
            endpoint = vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
            print(f"✅ Created endpoint: {endpoint_name}")
            return endpoint
        except Exception as e:
            raise Exception(f"Failed to create endpoint: {e}")

def check_table_exists(catalog: str, schema: str, table: str) -> bool:
    """
    Check if a table exists in the specified catalog and schema.
    Args:
        catalog (str): Catalog name.
        schema (str): Schema name.
        table (str): Table name.
    Returns:
        bool: True if table exists, False otherwise.
    """
    try:
        w = WorkspaceClient()
        w.tables.get(f"{catalog}.{schema}.{table}")
        return True
    except Exception:
        return False
    
def create_table_with_cdf_if_not_exists(spark, table_name):
    """Create a Delta table with Change Data Feed enabled if it does not exist."""
    try:
        spark.sql(f"DESCRIBE TABLE {table_name}")
        print(f"✅ Table already exists: {table_name}")
    except Exception as e:
        if "Table or view not found" not in str(e):
            print(f"⚠️ Error describing table {table_name}: {str(e)}")
        spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY,
            pdf_name STRING,
            content STRING,
            embedding ARRAY<FLOAT>,
            content_type STRING
        ) TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        print(f"✅ Created table: {table_name}")

def table_exists(catalog: str, schema: str, table: str) -> bool:
    """
    Alias for check_table_exists for clarity in notebook usage.
    Args:
        catalog (str): Catalog name.
        schema (str): Schema name.
        table (str): Table name.
    Returns:
        bool: True if table exists, False otherwise.
    """
    return check_table_exists(catalog, schema, table)

def copy_to_local(src_path: str) -> str:
    """
    Copy a remote file to a local temp file using Spark, or return as-is if already local.
    Args:
        src_path (str): Source file path (local or remote).
    Returns:
        str: Local file path.
    """
    import os
    import tempfile
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    if os.path.exists(src_path):
        return src_path
    local_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    with open(local_temp.name, "wb") as f:
        f.write(spark.read.format("binaryFile").load(src_path).collect()[0].content)
    return local_temp.name

def prepare_table_chunks(table_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare table chunks from extracted table documents (as-is, no chunking).
    Args:
        table_docs (List[Dict[str, Any]]): List of table document dicts.
    Returns:
        List[Dict[str, Any]]: List of table chunk dicts (same as input).
    """
    # For now, just return as-is (no chunking logic applied)
    return table_docs

def prepare_narrative_chunks(
    docs: List[Dict[str, Any]],
    chunk_size: int = 1024,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Chunk narrative documents into overlapping text chunks.
    Args:
        docs (List[Dict[str, Any]]): List of narrative document dicts with 'text' key.
        chunk_size (int): Max chunk size (characters).
        chunk_overlap (int): Overlap between chunks (characters).
    Returns:
        List[Dict[str, Any]]: List of chunked narrative dicts.
    """
    chunks = []
    for doc in docs:
        text = doc.get('text', '')
        meta = doc.get('meta', {})
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunk = {'text': chunk_text, 'meta': meta.copy()}
            chunks.append(chunk)
            if end == len(text):
                break
            start += chunk_size - chunk_overlap
    return chunks

def wait_for_index_deletion(vsc, vs_endpoint_name, index_name, timeout=600, poll_interval=15):
    """Wait until the index is fully deleted before proceeding."""
    start_time = time.time()
    while True:
        try:
            exists = index_exists(vsc, vs_endpoint_name, index_name)
            if not exists:
                print(f"✅ Index '{index_name}' is fully deleted.")
                break
            else:
                print(f"⏳ Waiting for index '{index_name}' to be deleted...")
        except Exception as e:
            print(f"⚠️  Error checking index status: {e}")
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout: Index '{index_name}' was not deleted after {timeout} seconds.")
        time.sleep(poll_interval)

def create_index_safely(
    vsc: VectorSearchClient,
    vs_endpoint_name: str,
    source_table_name: str,
    index_name: str,
    description: str,
    valid_tables: list
) -> bool:
    """
    Create a vector search index if it does not exist and sync if it does.
    Args:
        vsc (VectorSearchClient): The vector search client.
        vs_endpoint_name (str): The endpoint name.
        source_table_name (str): The source table name.
        index_name (str): The index name.
        description (str): Description for logging.
        valid_tables (list): List of valid table names.
    Returns:
        bool: True if index is ready or created, False otherwise.
    """
    table_short_name = source_table_name.split('.')[-1]
    if table_short_name not in valid_tables:
        print(f"⏭️  Skipping {description} - table missing")
        return False
    try:
        if not index_exists(vsc, vs_endpoint_name, index_name):
            print(f"📝 Creating {description}...")
            vsc.create_delta_sync_index(
                endpoint_name=vs_endpoint_name,
                index_name=index_name,
                source_table_name=source_table_name,
                pipeline_type="TRIGGERED",
                primary_key="id",
                embedding_dimension=1024,
                embedding_vector_column="embedding"
            )
            wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name)
            print(f"✅ {description} ready")
        else:
            print(f"♻️  {description} exists, syncing...")
            vsc.get_index(vs_endpoint_name, index_name).sync()
        return True
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False
    
def manage_index(vsc, vs_endpoint_name, source_table, index_name, description, force_rebuild, embedding_dimension, embedding_vector_column, embedding_model_endpoint_name):
    if force_rebuild:
        if index_exists(vsc, vs_endpoint_name, index_name):
            print(f"🗑️ Deleting existing {description}...")
            vsc.delete_index(vs_endpoint_name, index_name)
            wait_for_index_deletion(vsc, vs_endpoint_name, index_name)
        print(f"📝 Creating fresh {description} with dimension {embedding_dimension} and column '{embedding_vector_column}'...")
        vsc.create_delta_sync_index(
            endpoint_name=vs_endpoint_name,
            index_name=index_name,
            source_table_name=source_table,
            pipeline_type="TRIGGERED",
            primary_key="id",
            embedding_dimension=embedding_dimension,
            embedding_vector_column=embedding_vector_column,
            embedding_model_endpoint_name=embedding_model_endpoint_name
        )
        wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name)
        print(f"✅ Fresh {description} created and ready.")
    else:
        if index_exists(vsc, vs_endpoint_name, index_name):
            print(f"✅ {description} already exists.")
        else:
            print(f"❌ {description} does not exist. Set FORCE_REBUILD_INDICES = True to create it.")

def manage_index_text(vsc, vs_endpoint_name, source_table, index_name, description, force_rebuild, embedding_dimension, embedding_source_column, embedding_model_endpoint_name):
    if force_rebuild:
        if index_exists(vsc, vs_endpoint_name, index_name):
            print(f"🗑️ Deleting existing {description}...")
            vsc.delete_index(vs_endpoint_name, index_name)
            wait_for_index_deletion(vsc, vs_endpoint_name, index_name)
        print(f"📝 Creating fresh {description} with dimension {embedding_dimension} and raw text column '{embedding_source_column}'...")
        vsc.create_delta_sync_index(
            endpoint_name=vs_endpoint_name,
            index_name=index_name,
            source_table_name=source_table,
            pipeline_type="TRIGGERED",
            primary_key="id",
            embedding_dimension=embedding_dimension,
            embedding_source_column=embedding_source_column,
            embedding_model_endpoint_name=embedding_model_endpoint_name
        )
        wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name)
        print(f"✅ Fresh {description} created and ready.")
    else:
        if index_exists(vsc, vs_endpoint_name, index_name):
            print(f"✅ {description} already exists.")
        else:
            print(f"❌ {description} does not exist. Set FORCE_REBUILD_INDICES = True to create it.")

# --- Retriever and RAG Helpers ---
def get_index(
    vs_endpoint_name: str,
    vs_index_fullname: str
    ):
    """ 
    Get a self managed vector search index from the vector store.
    Args:
        vs_endpoint_name (str): Endpoint name.
        vs_index_fullname (str): Full index name for content.
        embedding_model: Embedding model instance.
    Returns:
        Index: Databricks Vector Search index instance.
    """
    vector_store = DatabricksVectorSearch(endpoint=vs_endpoint_name,
        index_name=vs_index_fullname, columns=["chunk_header"])
    
    return vector_store

def get_retriever(
    vs_endpoint_name: str,
    vs_index_fullname: str,
    embedding_model,
    k: int = 12
):
    """
    Get a retriever for content from the vector store.
    Args:
        vs_endpoint_name (str): Endpoint name.
        vs_index_fullname (str): Full index name for content.
        embedding_model: Embedding model instance.
        k (int): Number of results to retrieve.
    Returns:
        Retriever: LangChain retriever for content.
    """
    vector_store = DatabricksVectorSearch(endpoint_name = vs_endpoint_name,
        index_name=vs_index_fullname, text_column="content", embedding=embedding_model
    )
    return vector_store.as_retriever(search_kwargs={"k": k})

def get_enhanced_retriever(
    vs_endpoint_name: str,
    vs_index_fullname: str,
    k: int = 15,
):
    """
    Enhanced hybrid retriever using both headers and content..
    Args:
        vs_endpoint_name (str): Endpoint name.
        vs_index_fullname (str): Unified content index name.
        k (int): Number of results to retrieve.
    Returns:
        Callable: Function that retrieves passages for a query.
    """
    vector_store = get_index(vs_endpoint_name, vs_index_fullname)

    def retrieve(query: str, k_override: int = None) -> list:
        """
        Retrieve passages from the unified content index.
        Args:
            query (str): The search query.
            k_override (int, optional): Override for number of results.
        Returns:
            list: Retrieved passages with detected content types.
        """
        num_results = k_override if k_override is not None else k
        content_docs = vector_store.similarity_search(
            query=query,
            columns=["chunk_header","content"],  
            k=num_results,
            query_type="hybrid"
        )
        all_passages = []
        for doc in content_docs:
            content_text = doc.page_content
            all_passages.append({
                "text": content_text,
                "meta": {
                    **doc.metadata
                }
            })
        return all_passages if all_passages else []
    
    return retrieve

def get_narrative_retriever(
    vs_endpoint_name: str,
    narrative_vs_index_fullname: str,
    embedding_model: DatabricksEmbeddings,
    k: int = 3
):
    """
    Get a retriever for narrative content from the vector store.
    Args:
        vs_endpoint_name (str): Endpoint name.
        narrative_vs_index_fullname (str): Full index name for narrative content.
        embedding_model (DatabricksEmbeddings): Embedding model instance.
        k (int): Number of results to retrieve.
    Returns:
        Retriever: LangChain retriever for narrative content.
    """
    vsc = VectorSearchClient(disable_notice=True)
    vs_index = vsc.get_index(
        endpoint_name=vs_endpoint_name,
        index_name=narrative_vs_index_fullname,
    )
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})

def get_table_retriever(
    vs_endpoint_name: str,
    table_vs_index_fullname: str,
    embedding_model: DatabricksEmbeddings,
    k: int = 3
):
    """
    Get a retriever for table content from the vector store.
    Args:
        vs_endpoint_name (str): Endpoint name.
        table_vs_index_fullname (str): Full index name for table content.
        embedding_model (DatabricksEmbeddings): Embedding model instance.
        k (int): Number of results to retrieve.
    Returns:
        Retriever: LangChain retriever for table content.
    """
    vsc = VectorSearchClient(disable_notice=True)
    vs_index = vsc.get_index(
        endpoint_name=vs_endpoint_name,
        index_name=table_vs_index_fullname,
    )
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})

# --- FlashRank reranker for improved context relevance
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False
    print("⚠️ FlashRank not available - install with: pip install flashrank")

def rerank_passages(query: str, passages: list, top_k: int = 5) -> list:
    """
    Rerank passages using FlashRank and return the top_k most relevant.
    Args:
        query (str): The search query
        passages (list): List of passages, each with 'text' and 'meta' keys
        top_k (int): Number of top passages to return
    Returns:
        list: Reranked passages with scores
    """
    if not FLASHRANK_AVAILABLE:
        print("⚠️ FlashRank not available, returning original order")
        return passages[:top_k]
    
    try:
        ranker = Ranker(model_name="rank-T5-flan", cache_dir="/tmp/flashrank_cache")
        rerankrequest = RerankRequest(query=query, passages=passages)
        reranked_results = ranker.rerank(rerankrequest)
        return reranked_results[:top_k]
    except Exception as e:
        print(f"❌ Reranking failed: {e}, returning original order")
        return passages[:top_k]
    
def get_enhanced_retriever_with_reranking(
    vs_endpoint_name: str,
    content_vs_index_fullname: str,
    embedding_model,
    narrative_weight: float = 0.5,
    table_weight: float = 0.5,
    k: int = 6,
    rerank_top_k: int = 5
):
    """
    Enhanced retriever with FlashRank reranking for better context relevance using unified content index.
    Args:
        vs_endpoint_name (str): Endpoint name.
        content_vs_index_fullname (str): Unified content index name.
        embedding_model: Embedding model instance.
        k (int): Number of results to retrieve.
        rerank_top_k (int): Number of top reranked passages to return.
    Returns:
        Callable: Function that retrieves and reranks passages for a query.
    """
    content_retriever = get_retriever(vs_endpoint_name, content_vs_index_fullname, embedding_model, k)
    
    def retrieve(query: str) -> list:
        """
        Retrieve and rerank passages from the unified content index.
        Args:
            query (str): The search query.
        Returns:
            list: Reranked passages with detected content types.
        """
        # Get results from unified content retriever
        content_docs = content_retriever.invoke(query)
        
        # Format for reranking and detect content types
        all_passages = []
        
        for doc in content_docs:
            content_text = doc.page_content
            
            # Detect content type based on content characteristics
            numeric_ratio = sum(c.isdigit() or c in '$%,' for c in content_text) / len(content_text) if content_text else 0
            is_table_like = numeric_ratio > 0.1 or any(pattern in content_text for pattern in ['|', '\t', '  '])
            
            # Determine content type and weight
            content_type = "table" if is_table_like else "narrative"
            weight = table_weight if is_table_like else narrative_weight  
            
            all_passages.append({
                "text": content_text,
                "meta": {
                    "content_type": content_type,
                    "weight": weight,
                    "numeric_ratio": numeric_ratio,
                    "detected_type": content_type,
                    **doc.metadata
                }
            })
        
        # Rerank passages for better relevance
        if all_passages:
            reranked_passages = rerank_passages(query, all_passages, rerank_top_k)
            return reranked_passages
        else:
            return []
    
    return retrieve

def get_enhanced_hybrid_retriever_with_reranking(
    vs_endpoint_name: str,
    narrative_vs_index_fullname: str,
    table_vs_index_fullname: str,
    embedding_model: DatabricksEmbeddings,
    narrative_weight: float = 0.6,
    table_weight: float = 0.4,
    k: int = 3,
    rerank_top_k: int = 5
):
    """
    Enhanced hybrid retriever with FlashRank reranking for better context relevance.
    Args:
        vs_endpoint_name (str): Endpoint name.
        narrative_vs_index_fullname (str): Narrative index name.
        table_vs_index_fullname (str): Table index name.
        embedding_model (DatabricksEmbeddings): Embedding model instance.
        narrative_weight (float): Weight for narrative passages.
        table_weight (float): Weight for table passages.
        k (int): Number of results per retriever.
        rerank_top_k (int): Number of top reranked passages to return.
    Returns:
        Callable: Function that retrieves and reranks passages for a query.
    """
    narrative_retriever = get_narrative_retriever(vs_endpoint_name, narrative_vs_index_fullname, embedding_model, k)
    table_retriever = get_table_retriever(vs_endpoint_name, table_vs_index_fullname, embedding_model, k)
    
    def retrieve(query: str) -> list:
        """
        Retrieve and rerank passages from both narrative and table retrievers.
        Args:
            query (str): The search query.
        Returns:
            list: Reranked passages.
        """
        # Get results from both retrievers
        narrative_docs = narrative_retriever.invoke(query)
        table_docs = table_retriever.invoke(query)
        
        # Format for reranking
        all_passages = []
        
        # Add narrative documents
        for doc in narrative_docs:
            all_passages.append({
                "text": doc.page_content,
                "meta": {
                    "content_type": "narrative",
                    "weight": narrative_weight,
                    **doc.metadata
                }
            })
        
        # Add table documents
        for doc in table_docs:
            all_passages.append({
                "text": doc.page_content,
                "meta": {
                    "content_type": "table", 
                    "weight": table_weight,
                    **doc.metadata
                }
            })
        
        # Rerank passages for better relevance
        if all_passages:
            reranked_passages = rerank_passages(query, all_passages, rerank_top_k)
            return reranked_passages
        else:
            return []
    
    return retrieve

# Enhanced RetrievalQA with reranking and unified content index
class EnhancedRetrievalQA:
    """
    Enhanced Retrieval QA system with reranking and Llama-4 optimized prompt for annual report analysis using unified content index.
    """
    def __init__(
        self,
        llm,
        enhanced_retriever,
        enhanced_retriever_with_reranking=None,
        custom_prompt_template: PromptTemplate = None,
        reranking: bool = False
    ):
        self.llm = llm
        self.reranking = reranking
        if reranking:
            if enhanced_retriever_with_reranking is None:
                raise ValueError("enhanced_retriever_with_reranking must be provided when reranking=True")
            self.content_retriever = enhanced_retriever_with_reranking
        else:
            self.content_retriever = enhanced_retriever
        
        # Llama-4 optimized prompt template for annual report analysis with unified content
        self.prompt_template = custom_prompt_template or PromptTemplate(
            template="""You are FinanceGPT, a highly capable financial analysis assistant. You specialize in analyzing annual reports and extracting key financial insights to help users make informed decisions.

Your expertise includes:
- Financial statement analysis and ratio interpretation
- Revenue and profitability trend identification  
- Risk assessment and strategic positioning analysis
- Market performance and competitive landscape evaluation
- Regulatory compliance and governance insights

CONTEXT INFORMATION:
You have access to retrieved information from annual reports that includes both narrative and tabular content:
- [NARRATIVE]: Management discussion, strategic insights, qualitative explanations, and contextual information
- [TABLE]: Financial data, metrics, ratios, and structured quantitative information
- [DETECTED]: Content type automatically detected based on structure and content characteristics

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

RESPONSE GUIDELINES:
1. **Accuracy**: Base your response strictly on the provided context. Do not fabricate information.
2. **Source Attribution**: Clearly indicate when information comes from narrative or tabular sources.
3. **Comprehensive Analysis**: Synthesize both narrative and tabular data for complete insights.
4. **Financial Expertise**: Apply financial analysis principles and provide contextual interpretation.
5. **Clarity**: Use clear, professional language suitable for business stakeholders.
6. **Content Synthesis**: Leverage the unified content structure to provide holistic insights.
7. **Limitations**: If the context lacks sufficient information, state this explicitly.

RESPONSE FORMAT:
- Start with a direct answer to the question
- Provide supporting details from the context, noting content types
- Include relevant financial metrics or data points when available
- Highlight insights that combine narrative and quantitative information
- Conclude with any important caveats or additional context

Your Response:""",
            input_variables=["context", "question"]
        )
    
    def __call__(self, query: str) -> dict:
        """
        Run the enhanced retrieval QA pipeline for a given query.
        Args:
            query (str): The user query.
        Returns:
            dict: QA result with response, sources, and detected content types.
        """
        # Retrieve and rerank relevant passages
        passages = self.content_retriever(query)
        
        # Prepare context with clear source attribution and content type detection
        context_parts = []
        
        for i, passage in enumerate(passages):
            score = passage.get('score', 0.0)
            
            context_parts.append(
                f"(Relevance: {score:.3f}): {passage['text']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Generate response using formatted prompt
        formatted_prompt = self.prompt_template.format(context=context, question=query)
        response = self.llm.invoke(formatted_prompt)
        
        return {
            'query': query,
            'result': response.content if hasattr(response, 'content') else str(response),
            'source_documents': passages,
            'reranked': True if self.reranking else False,
            'num_sources': len(passages),
            'unified_index': True
        }

class EnhancedHybridRetrievalQA:
    """
    Enhanced Retrieval QA system with reranking and optimized prompt for annual report analysis using hybrid content index.
    """
    def __init__(
        self,
        llm,
        enhanced_hybrid_retriever_with_reranking,
        custom_prompt_template: PromptTemplate = None
    ):
        """
        Initialize the EnhancedRetrievalQA system.
        Args:
            llm: The language model to use for generation.
            enhanced_retriever_with_reranking: Callable retriever with reranking.
            custom_prompt_template (PromptTemplate, optional): Custom prompt template.
        """
        self.llm = llm
        self.content_retriever = enhanced_hybrid_retriever_with_reranking
        
        # Llama-4 optimized prompt template for annual report analysis with unified content
        self.prompt_template = custom_prompt_template or PromptTemplate(
            template="""You are FinanceGPT, a highly capable financial analysis assistant. You specialize in analyzing annual reports and extracting key financial insights to help users make informed decisions.

Your expertise includes:
- Financial statement analysis and ratio interpretation
- Revenue and profitability trend identification  
- Risk assessment and strategic positioning analysis
- Market performance and competitive landscape evaluation
- Regulatory compliance and governance insights

CONTEXT INFORMATION:
You have access to retrieved information from annual reports that includes both narrative and tabular content:
- [NARRATIVE]: Management discussion, strategic insights, qualitative explanations, and contextual information
- [TABLE]: Financial data, metrics, ratios, and structured quantitative information
- [DETECTED]: Content type automatically detected based on structure and content characteristics

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

RESPONSE GUIDELINES:
1. **Accuracy**: Base your response strictly on the provided context. Do not fabricate information.
2. **Source Attribution**: Clearly indicate when information comes from narrative or tabular sources.
3. **Comprehensive Analysis**: Synthesize both narrative and tabular data for complete insights.
4. **Financial Expertise**: Apply financial analysis principles and provide contextual interpretation.
5. **Clarity**: Use clear, professional language suitable for business stakeholders.
6. **Content Synthesis**: Leverage the unified content structure to provide holistic insights.
7. **Limitations**: If the context lacks sufficient information, state this explicitly.

RESPONSE FORMAT:
- Start with a direct answer to the question
- Provide supporting details from the context, noting content types
- Include relevant financial metrics or data points when available
- Highlight insights that combine narrative and quantitative information
- Conclude with any important caveats or additional context

Your Response:""",
            input_variables=["context", "question"]
        )
    
    def __call__(self, query: str) -> dict:
        """
        Run the enhanced retrieval QA pipeline for a given query.
        Args:
            query (str): The user query.
        Returns:
            dict: QA result with response, sources, and detected content types.
        """
        # Retrieve and rerank relevant passages
        reranked_passages = self.content_retriever(query)
        
        # Prepare context with clear source attribution and content type detection
        context_parts = []
        context_types = []
        narrative_count = 0
        table_count = 0
        
        for i, passage in enumerate(reranked_passages):
            detected_type = passage['meta'].get('detected_type', 'unknown')
            content_type = passage['meta'].get('content_type', detected_type)
            score = passage.get('score', 0.0)
            numeric_ratio = passage['meta'].get('numeric_ratio', 0.0)
            
            context_types.append(content_type)
            
            # Count content types
            if content_type == 'narrative':
                narrative_count += 1
            elif content_type == 'table':
                table_count += 1
            
            # Enhanced context formatting with detection info
            type_indicator = f"[{content_type.upper()}]"
            if content_type == 'table':
                type_indicator += f" (Numeric: {numeric_ratio:.2f})"
            
            context_parts.append(
                f"{type_indicator} (Relevance: {score:.3f}): {passage['text']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Generate response using formatted prompt
        formatted_prompt = self.prompt_template.format(context=context, question=query)
        response = self.llm.invoke(formatted_prompt)
        
        return {
            'query': query,
            'result': response.content if hasattr(response, 'content') else str(response),
            'source_documents': reranked_passages,
            'context_types': context_types,
            'content_breakdown': {
                'narrative': narrative_count,
                'table': table_count,
                'total': len(reranked_passages)
            },
            'reranked': True,
            'num_sources': len(reranked_passages),
            'unified_index': True
        }