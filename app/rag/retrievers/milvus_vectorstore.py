import warnings
from typing import List

from langchain.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.docstore.document import Document
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from app.rag.retrievers.base import BaseRetrieverService


class MilvusRetriever(VectorStoreRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            return self.vectorstore.similarity_search(query, **self.search_kwargs)

        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = self.vectorstore.similarity_search_with_score(
                query, **self.search_kwargs
            )
            score_threshold = self.search_kwargs.get("score_threshold")

            # warn if similarity not in [0, 1]
            if any(s < 0.0 or s > 1.0 for _, s in docs_and_similarities):
                warnings.warn(f"Relevance scores out of range: {docs_and_similarities}")

            filtered = [
                (doc, s)
                for doc, s in docs_and_similarities
                if score_threshold is None or s >= score_threshold
            ]

            if not filtered:
                warnings.warn(f"No relevant docs above threshold {score_threshold}")

            return [doc for doc, _ in filtered]

        elif self.search_type == "mmr":
            return self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )

        else:
            raise ValueError(f"Invalid search_type: {self.search_type}")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            return await self.vectorstore.asimilarity_search(query, **self.search_kwargs)

        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = await self.vectorstore.asimilarity_search_with_score(
                query, **self.search_kwargs
            )
            score_threshold = self.search_kwargs.get("score_threshold")

            filtered = [
                (doc, s)
                for doc, s in docs_and_similarities
                if score_threshold is None or s >= score_threshold
            ]

            if not filtered:
                warnings.warn(f"No relevant docs above threshold {score_threshold}")

            return [doc for doc, _ in filtered]

        elif self.search_type == "mmr":
            return await self.vectorstore.amax_marginal_relevance_search(
                query, **self.search_kwargs
            )

        else:
            raise ValueError(f"Invalid search_type: {self.search_type}")


class MilvusVectorstoreRetrieverService(BaseRetrieverService):
    def do_init(
        self,
        retriever: BaseRetriever = None,
        top_k: int = 5,
    ):
        self.vs = None
        self.top_k = top_k
        self.retriever = retriever

    @classmethod
    def from_vectorstore(
        cls,
        vectorstore: VectorStore,
        top_k: int,
        score_threshold: int | float,
    ):
        retriever = MilvusRetriever(
            vectorstore=vectorstore,
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": top_k},
        )
        return cls(retriever=retriever, top_k=top_k)

    def get_relevant_documents(self, query: str):
        return self.retriever.get_relevant_documents(query)
