import os
import yaml
import base64

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

from helper.functions import (
    is_databricks_environment,
    create_vector_search_endpoint,
    manage_index_text
    )

vs_endpoint_prefix = "vs_endpoint_"
vs_endpoint_name = vs_endpoint_prefix + "poc"

embedding_table_name = "<table_name>" #delta table created in parser_chunker.py
vs_index_name = "<index_name>"

# Databricks Authentication Setup

# Initialize VectorSearchClient
vsc = None
auth_success = False

if is_databricks_environment():
    print("✅ Databricks environment detected")
    try:
        vsc = VectorSearchClient(disable_notice=True)
        endpoints = vsc.list_endpoints()
        print(f"✅ Authentication successful ({len(endpoints.get('endpoints', []))} endpoints)")
        auth_success = True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")

else:
    # Try secrets first, with fallback to config file
    try:
        # Clear environment variables
        for var in ["DATABRICKS_AUTH_TYPE", "DATABRICKS_CLUSTER_ID", "DATABRICKS_METADATA_SERVICE_URL", "DATABRICKS_HOST", "DATABRICKS_TOKEN"]:
            if var in os.environ:
                del os.environ[var]
        
        # First attempt: Get credentials from secrets
        try:
            w = WorkspaceClient()
            databricks_host_raw = w.secrets.get_secret(scope="rag-secrets", key="databricks-host").value
            databricks_token_raw = w.secrets.get_secret(scope="rag-secrets", key="databricks-token").value
            
            # Decode if base64 encoded
            try:
                DATABRICKS_HOST = base64.b64decode(databricks_host_raw).decode('utf-8').rstrip('/')
            except:
                DATABRICKS_HOST = databricks_host_raw.rstrip('/')
                
            try:
                DATABRICKS_TOKEN = base64.b64decode(databricks_token_raw).decode('utf-8')
            except:
                DATABRICKS_TOKEN = databricks_token_raw
            
            print("✅ Using credentials from Databricks secrets")
            
        except Exception as secrets_error:
            print(f"⚠️  Secrets access failed: {secrets_error}")
            print("🔄 Falling back to config file credentials")
            
            # Fallback: Load from config file
            with open("../../config/training_config.yaml", "r") as file:
                config = yaml.safe_load(file)
            
            if "auth" in config and "databricks_host" in config["auth"] and "databricks_token" in config["auth"]:
                DATABRICKS_HOST = config["auth"]["databricks_host"].rstrip('/')
                DATABRICKS_TOKEN = config["auth"]["databricks_token"]
                print("✅ Using credentials from config file")
            else:
                raise Exception("No authentication credentials found in config file")
        
    except Exception as e:
        print(f"❌ Authentication setup failed: {e}")
        raise Exception("Authentication failed - check secrets or config file credentials")
    
    # Clear and set final environment variables
    for var in ["DATABRICKS_AUTH_TYPE", "DATABRICKS_CLUSTER_ID", "DATABRICKS_METADATA_SERVICE_URL", "DATABRICKS_HOST", "DATABRICKS_TOKEN"]:
        if var in os.environ:
            del os.environ[var]
            
    os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
    os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
    os.environ["DATABRICKS_AUTH_TYPE"] = "pat"
    
    # Initialize VectorSearchClient
    try:
        vsc = VectorSearchClient(
            disable_notice=True,
            workspace_url=DATABRICKS_HOST,
            personal_access_token=DATABRICKS_TOKEN
        )
        
        # Test connection
        endpoints = vsc.list_endpoints()
        print(f"✅ Authenticated successfully ({len(endpoints.get('endpoints', []))} endpoints)")
        auth_success = True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")

if not auth_success:
    raise Exception("Authentication failed - check token validity")

# Configuration and Setup
with open("../../config/training_config.yaml", "r") as file:
    config = yaml.safe_load(file)

catalog_name = config["data"]["catalog_name"]
schema_name = config["data"]["schema_name"]

print(f"📋 Config: {catalog_name}.{schema_name} | Endpoint: {vs_endpoint_name}")

# Vector Search Endpoint Setup
endpoint = create_vector_search_endpoint(vsc, vs_endpoint_name)

embedding_provider = "openai"

# Option to force rebuild indices (recommended after data preparation)

FORCE_REBUILD_INDICES = True  # Set to True to ensure fresh data

content_source = f"{catalog_name}.{schema_name}.{embedding_table_name}"
content_index = f"{catalog_name}.{schema_name}.{vs_index_name}"

# Set embedding parameters based on provider
if embedding_provider == "databricks":
    embedding_dimension = 1024
    embedding_vector_column = "embedding"
    embedding_source_column = "content"
    embedding_model_endpoint_name = config["endpoint"]["databricks_embeddings_model_endpoint"]
elif embedding_provider == "openai":
    embedding_dimension = 3072
    embedding_vector_column = "openai_embedding"
    embedding_source_column = "content"
    embedding_model_endpoint_name = config["endpoint"]["openai_embeddings_model_endpoint"]
else:
    raise ValueError(f"Unknown embedding provider: {embedding_provider}")

# Manage the content index
manage_index_text(vsc, vs_endpoint_name, content_source, content_index, "Content Index", FORCE_REBUILD_INDICES, embedding_dimension, embedding_source_column, embedding_model_endpoint_name)