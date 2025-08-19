import os
import yaml
import pandas as pd

from databricks.connect import DatabricksSession
from delta.tables import DeltaTable
from pyspark.sql.functions import pandas_udf, col
from mlflow.deployments import get_deploy_client

import openai
import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

from helper.custom_chunker import CustomHybridChunker
from helper.functions import table_exists, copy_to_local

import warnings
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.")


# Load Configuration
import yaml
with open("../../config/training_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Databricks Connection Setup
print("🔍 Setting up Databricks remote Spark session...")
spark = DatabricksSession.builder.getOrCreate()
print("✅ Spark session ready!")

embedding_table_name = "<table_name>"

file_path = config["data"]["file_path"]
catalog_name = config["data"]["catalog_name"]
schema_name = config["data"]["schema_name"]

print(f"📋 Configuration loaded: {file_path}, {catalog_name}, {schema_name}")

embedding_table_full_name = f"{catalog_name}.{schema_name}.{embedding_table_name}"

# Create and Upsert PDF Metadata into pdf_annual_report (no duplicates, table is preserved)
pdf_metadata_table = f"{catalog_name}.{schema_name}.pdf_annual_report"
spark.sql(f'''
    CREATE TABLE IF NOT EXISTS {pdf_metadata_table} (
        file_name STRING,
        file_path STRING,
        file_size LONG,
        modification_time TIMESTAMP
    ) TBLPROPERTIES (delta.enableChangeDataFeed = true)
''')
pdf_files_df = spark.read.format("binaryFile").option("recursiveFileLookup", "true").load(file_path)
pdf_metadata_df = pdf_files_df.selectExpr(
    "element_at(split(path, '/'), -1) as file_name",
    "path as file_path",
    "length as file_size",
    "modificationTime as modification_time"
).where("lower(file_name) LIKE '%.pdf'")

try:
    delta_tbl = DeltaTable.forName(spark, pdf_metadata_table)
    delta_tbl.alias("t").merge(
        pdf_metadata_df.alias("s"),
        "t.file_path = s.file_path"
    ).whenMatchedUpdateAll() \
     .whenNotMatchedInsertAll() \
     .execute()
    print(f"✅ Upserted PDF metadata into table: {pdf_metadata_table} ({pdf_metadata_df.count()} files scanned)")
except Exception as e:
    pdf_metadata_df.write.mode("append").saveAsTable(pdf_metadata_table)
    print(f"⚠️ Merge failed, fallback to append. {str(e)}")

pdf_files_df = spark.read.format("binaryFile").option("recursiveFileLookup", "true").load(file_path)
pdf_files = [row.path for row in pdf_files_df.select("path").collect() if row.path.lower().endswith(".pdf")]

class MDTableSerializerProvider(ChunkingSerializerProvider):
                def get_serializer(self, doc):
                    return ChunkingDocSerializer(
                        doc=doc,
                        table_serializer=MarkdownTableSerializer(),  
                    )

if not table_exists(catalog_name, schema_name, embedding_table_name):
    for pdf in pdf_files:
        try:
            local_pdf = copy_to_local(pdf)
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options.use_gpu = False  # <-- set this.
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True

            doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
                )
            doc = doc_converter.convert(source=local_pdf).document

            tokenizer = OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model("gpt-4-1-mini"),
            max_tokens=1_000_000,  # context window length required for OpenAI tokenizers
                )
                
            chunker = CustomHybridChunker(
                tokenizer=tokenizer,
                serializer_provider=MDTableSerializerProvider(),
                )
            
            chunk_iter = chunker.chunk(dl_doc=doc)
            chunks = list(chunk_iter)
            # Clean up temp file if it was created
            if local_pdf != pdf and os.path.exists(local_pdf):
                os.remove(local_pdf)
        except Exception as e:
            print(f"⚠️ Skipping {pdf}: {e}")
    print(f"✅ Extracted {len(chunks)} documents using Docling Parse with EasyOCR (CPU only).")
else:
    print(f"✅ {embedding_table_name} already exist, skipping Docling extraction.")

if not table_exists(catalog_name, schema_name, embedding_table_name):
    import pandas as pd

    data = []
    for n, chunk in enumerate(chunks):
        ctx_text = chunker.contextualize(chunk=chunk)
        num_tokens = tokenizer.count_tokens(text=ctx_text)
        table_ids = [int(item.self_ref.split('/')[-1]) for item in chunk.meta.doc_items if item.label.name == "TABLE"]
        text_ids = [int(item.self_ref.split('/')[-1]) for item in chunk.meta.doc_items if item.label.name == "TEXT"]
        page_numbers = list({item[0].page_no for item in [it.prov for it in chunk.meta.doc_items]})
        chunk_header = ", ".join(chunk.meta.headings or [])
        data.append({
            "id": n,
            "num_tokens": num_tokens,
            "table_ids": table_ids,
            "text_ids": text_ids,
            "page_numbers": page_numbers,
            "chunk_header": chunk_header,
            "content": ctx_text
        })

    df = pd.DataFrame(data)
else:
    print(f"✅ {embedding_table_name} already exist, skipping DataFrame creation.")

if not table_exists(catalog_name, schema_name, embedding_table_name):
    spark_df = spark.createDataFrame(df)
else:
    spark_narrative_df = None
    print(f"✅ {embedding_table_name} already exist, skipping Spark narrative DataFrame creation.")


embeddings_model_endpoint = config["endpoint"]["databricks_embeddings_model_endpoint"]
@pandas_udf("array<float>")
def get_embedding(content: pd.Series) -> pd.Series:
    deploy_client = get_deploy_client("databricks")
    def get_embeddings(batch):
        try:
            response = deploy_client.predict(
                endpoint=embeddings_model_endpoint,
                inputs={"input": batch}
            )
            return [e['embedding'] for e in response["data"]]
        except Exception as e:
            print(f"Embedding error for batch: {e}")
            return [[0.0] * 1024 for _ in batch]
    max_batch_size = 150
    batches = [content.iloc[i:i + max_batch_size] for i in range(0, len(content), max_batch_size)]
    all_embeddings = []
    for batch in batches:
        batch_embeddings = get_embeddings(batch.tolist())
        all_embeddings.extend(batch_embeddings)
    return pd.Series(all_embeddings)
print("✅ Embedding generation UDF defined")

openai_api_key = config["auth"].get("openai_api_key", None)
openai_embedding_model = "text-embedding-3-large"

if not openai_api_key:
    raise ValueError("OpenAI API key is missing. Please set 'openai_api_key' in your config.")

os.environ["OPENAI_API_KEY"] = openai_api_key

@pandas_udf("array<float>")
def get_openai_embedding(content: pd.Series) -> pd.Series:
    client = openai.OpenAI(api_key=openai_api_key)
    def get_embeddings(batch):
        try:
            response = client.embeddings.create(input=batch, model=openai_embedding_model)
            return [e.embedding for e in response.data]
        except Exception as e:
            print(f"OpenAI embedding error for batch: {e}")
            return [[0.0] * 3072 for _ in batch]
    max_batch_size = 50
    batches = [content.iloc[i:i + max_batch_size] for i in range(0, len(content), max_batch_size)]
    all_embeddings = []
    for batch in batches:
        batch_embeddings = get_embeddings(batch.tolist())
        all_embeddings.extend(batch_embeddings)
    return pd.Series(all_embeddings)

print("✅ OpenAI embedding generation UDF defined")

# Add OpenAI and Databricks embeddings to Spark DataFrames and save to Delta tables

if not table_exists(catalog_name, schema_name, embedding_table_name)  and spark_df is not None:
    print("🚀 Generating and saving OpenAI and Databricks embeddings...")
    content_embedded = spark_df.withColumn("embedding", get_embedding(col("content")))\
        .withColumn("openai_embedding", get_openai_embedding(col("content")))
    content_embedded.write.mode("append").saveAsTable(embedding_table_full_name)
    print(f"✅ Saved embeddings to: {embedding_table_full_name}")
else:
    print(f"✅ {embedding_table_full_name} already exists and is not empty, skipping embedding generation and save.")

# Enable Change Data Feed (CDF) after table creation and save
try:
    spark.sql(f"ALTER TABLE {embedding_table_full_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"✅ Enabled Change Data Feed on {embedding_table_full_name}")
except Exception as e:
    print(f"⚠️ Could not enable CDF on {embedding_table_full_name}: {e}")