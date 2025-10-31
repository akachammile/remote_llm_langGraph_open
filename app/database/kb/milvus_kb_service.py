import os
from langchain.schema import Document
from typing import Any, Dict, List, Optional
from app.cores.config import config
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus

from app.rag.retrievers.milvus_vectorstore import  MilvusVectorstoreRetrieverService


class MilvusKBService:
    milvus: Milvus  # milvus实体

    def __init__(self):
        self.kb_name = "test_kb_1"
        self.embed_model = config.EMBEDING_MODEL
        self.do_init()

    @staticmethod
    def get_collection(milvus_name):
        from pymilvus import Collection

        return Collection(name=milvus_name)

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        result = []
        if self.milvus.col:
            data_list = self.milvus.col.query(
                expr=f"pk in {[int(_id) for _id in ids]}", output_fields=["*"]
            )
            for data in data_list:
                text = data.pop("text")
                result.append(Document(page_content=text, metadata=data))
        return result

    @staticmethod
    def search(milvus_name, content, limit=3):
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        c = MilvusKBService.get_collection(milvus_name)
        return c.search(
            content, "embeddings", search_params, limit=limit, output_fields=["content"]
        )

    def do_create_kb(self) -> None:
        pass

    # def vs_type(self) -> str:
    #     return SupportedVSType.MILVUS

    def get_Embeddings(self, model_name: str):
        embeding_model = OpenAIEmbeddings(
            model=model_name, base_url=config.MODEL_BASE_URL, api_key="ollama"
        )
        return embeding_model

    def _load_milvus(self):
        self.milvus = Milvus(
            embedding_function=self.get_Embeddings(self.embed_model),
            collection_name=self.kb_name,
            connection_args={
                "host": "172.18.0.207",
                "port": "19530",
                "user": "",
                "password": "",
                "secure": False,
            },
            index_params={"metric_type": "L2", "index_type": "HNSW"},
            search_params={"metric_type": "L2"},
            auto_id=True,
        )

    def do_init(self):
        self._load_milvus()

    def do_drop_kb(self):
        if self.milvus.col:
            self.milvus.col.release()
            self.milvus.col.drop()

    def do_search(self, query: str, top_k: int, score_threshold: float):
        self._load_milvus()
        # embed_func = get_Embeddings(self.embed_model)
        # embeddings = embed_func.embed_query(query)
        # docs = self.milvus.similarity_search_with_score_by_vector(embeddings, top_k)
        retriever = MilvusVectorstoreRetrieverService.from_vectorstore(
            self.milvus,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        docs = retriever.get_relevant_documents(query)
        return docs

    def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
        for doc in docs:
            for k, v in doc.metadata.items():
                doc.metadata[k] = str(v)
            for field in self.milvus.fields:
                doc.metadata.setdefault(field, "")
            doc.metadata.pop(self.milvus._text_field, None)
            doc.metadata.pop(self.milvus._vector_field, None)

        ids = self.milvus.add_documents(docs)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        return doc_infos

    # def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
    #     id_list = list_file_num_docs_id_by_kb_name_and_file_name(
    #         kb_file.kb_name, kb_file.filename
    #     )
    #     if self.milvus.col:
    #         self.milvus.col.delete(expr=f"pk in {id_list}")

    # Issue 2846, for windows
    # if self.milvus.col:
    #     file_path = kb_file.filepath.replace("\\", "\\\\")
    #     file_name = os.path.basename(file_path)
    #     id_list = [item.get("pk") for item in
    #                self.milvus.col.query(expr=f'source == "{file_name}"', output_fields=["pk"])]
    #     self.milvus.col.delete(expr=f'pk in {id_list}')

    def do_clear_vs(self):
        if self.milvus.col:
            self.do_drop_kb()
            self.do_init()


# if __name__ == "__main__":
# # 测试建表使用
# from chatchat.server.db.base import Base, engine

# Base.metadata.create_all(bind=engine)
# milvusService = MilvusKBService("test")
# # milvusService.add_doc(KnowledgeFile("README.md", "test"))

# print(milvusService.get_doc_by_ids(["444022434274215486"]))
# # milvusService.delete_doc(KnowledgeFile("README.md", "test"))
# # milvusService.do_drop_kb()
# # print(milvusService.search_docs("如何启动api服务"))
