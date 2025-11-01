import os
from typing import Any, Dict, List
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from pymilvus import Collection, connections, utility
from app.cores.config import config
from app.logger import logger
from app.rag.retrievers.milvus_vectorstore import MilvusVectorstoreRetrieverService


class MilvusKBService:
    milvus: Milvus

    def __init__(self):
        self.kb_name = "your_collection_name"
        self.embed_model = config.EMBEDING_MODEL or "dengcao/Qwen3-Embedding-8B:Q5_K_M"
        self.base_url = config.EMBEDING_MODEL_BASE_URL or "http://localhost:11434"
        self.milvus_host = "172.18.0.207"
        self.milvus_port = "19530"

        logger.info(f"🔹 Using embedding model: {self.embed_model}")
        self._connect_milvus()
        self.do_init()

    # -----------------------
    # Milvus Connection
    # -----------------------
    def _connect_milvus(self):
        """确保 pymilvus 连接存在"""
        alias = "default"
        if not connections.has_connection(alias):
            logger.info(f"🔗 Connecting pymilvus to {self.milvus_host}:{self.milvus_port}")
            connections.connect(
                alias=alias,
                uri=f"http://{self.milvus_host}:{self.milvus_port}"
            )

    # -----------------------
    # Embedding function
    # -----------------------
    def get_embeddings(self):
        return OllamaEmbeddings(
            model=self.embed_model,
            base_url=self.base_url,
        )

    # -----------------------
    # Milvus Initialization
    # -----------------------
    def _load_milvus(self):
        logger.info("🔹 Loading Milvus vector store...")
        self.milvus = Milvus(
            embedding_function=self.get_embeddings(),
            collection_name=self.kb_name,
            connection_args={
                "uri": f"http://{self.milvus_host}:{self.milvus_port}",
            },
            index_params={
                "metric_type": "L2",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            },
            search_params={
                "metric_type": "L2",
                "params": {"ef": 50},
            },
            auto_id=True,
        )
        self._ensure_collection_and_index()

    def _ensure_collection_and_index(self):
        """手动确保 collection 和 index 存在"""
        try:
            if not utility.has_collection(self.kb_name):
                logger.info(f"🆕 Creating new collection {self.kb_name}...")
                # 使用空文档触发 schema 创建
                self.milvus.add_documents([])

            collection = Collection(self.kb_name)
            if not collection.indexes:
                logger.info("⚙️ Creating HNSW index...")
                collection.create_index(
                    field_name="vector",
                    index_params={
                        "metric_type": "L2",
                        "index_type": "HNSW",
                        "params": {"M": 16, "efConstruction": 200},
                    },
                )
            collection.load()
            logger.info(f"✅ Collection {self.kb_name} is ready.")
        except Exception as e:
            logger.error(f"❌ Error ensuring collection/index: {e}")

    # -----------------------
    # Public Methods
    # -----------------------
    def do_init(self):
        self._load_milvus()

    def do_drop_kb(self):
        """删除 collection"""
        if utility.has_collection(self.kb_name):
            logger.warning(f"⚠️ Dropping collection {self.kb_name}")
            col = Collection(self.kb_name)
            col.release()
            col.drop()

    def do_clear_vs(self):
        """清空向量数据库"""
        self.do_drop_kb()
        self.do_init()

    def do_add_doc(self, docs: List[Document]) -> List[Dict]:
        logger.info(f"📥 Adding {len(docs)} docs to {self.kb_name}")
        # 确保 metadata 为字符串
        for doc in docs:
            for k, v in doc.metadata.items():
                doc.metadata[k] = str(v)
        ids = self.milvus.add_documents(docs)
        return [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]

    def do_search(self, query: str, top_k: int = 3, score_threshold: float = 0.5):
        retriever = MilvusVectorstoreRetrieverService.from_vectorstore(
            self.milvus,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        return retriever.get_relevant_documents(query)

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        if not utility.has_collection(self.kb_name):
            return []
        collection = Collection(self.kb_name)
        data_list = collection.query(
            expr=f"pk in {[int(_id) for _id in ids]}",
            output_fields=["*"],
        )
        result = []
        for data in data_list:
            text = data.pop("text", "")
            result.append(Document(page_content=text, metadata=data))
        return result
