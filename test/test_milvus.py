from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from pymilvus import utility, connections
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 把项目根目录加入sys.path
from app.cores.config import config
def get_Embeddings(model_name: str):
        embeding_model = OllamaEmbeddings(
            model=model_name, base_url=config.EMBEDING_MODEL_BASE_URL
        )
        return embeding_model
    
embeddings =get_Embeddings(model_name="dengcao/Qwen3-Embedding-8B:Q5_K_M")

# 尝试连接 Milvus
vector_store = Milvus(
            embedding_function=embeddings,
            collection_name="x",
            connection_args={
                "uri": "tcp://172.18.0.207",
                "port": "19530",
            },
            index_params={"metric_type": "L2", "index_type": "HNSW"},
            search_params={"metric_type": "L2"},
            auto_id=True,
        )

# 测试是否连通
print("✅ Connected Milvus successfully!")

# 验证 collection 是否存在
print("Collections:", utility.list_collections())
