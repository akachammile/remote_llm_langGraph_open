import os
import json
import traceback
import chardet
import importlib
from pathlib import Path
from app.logger import logger
from functools import lru_cache
from urllib.parse import urlencode
from app.cores.config import config
from langchain.text_splitter import TextSplitter
from langchain.docstore.document import Document
from typing import Dict, List, Union, Generator, Tuple, Optional
from app.rag.file_splitter import zh_title_enhance as func_zh_title_enhance
from app.tools.utils import thread_pool_executor
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader

LOADER_DICT = {
    # "UnstructuredHTMLLoader": [".html", ".htm"],
    # "MHTMLLoader": [".mhtml"],
    # "TextLoader": [".md"],
    # "UnstructuredMarkdownLoader": [".md"],
    # "JSONLoader": [".json"],
    # "JSONLinesLoader": [".jsonl"],
    "CSVLoader": [".csv"],
    # "FilteredCSVLoader": [".csv"], 如果使用自定义分割csv
    "PDFLoader": [".pdf"],
    "DocLoader": [".docx"],
    "PPTLoader": [
        ".ppt",
        ".pptx",
    ],
    "RapidOCRLoader": [".png", ".jpg", ".jpeg", ".bmp", ".tif"],
    # "UnstructuredFileLoader": [
    #     ".eml",
    #     ".msg",
    #     ".rst",
    #     ".rtf",
    #     ".txt",
    #     ".xml",
    #     ".epub",
    #     ".odt",
    #     ".tsv",
    # ],
    # "UnstructuredEmailLoader": [".eml", ".msg"],
    # "UnstructuredEPubLoader": [".epub"],
    "UnstructuredExcelLoader": [".xlsx", ".xls", ".xlsd"],
    # "NotebookLoader": [".ipynb"],
    # "UnstructuredODTLoader": [".odt"],
    # "PythonLoader": [".py"],
    # "UnstructuredRSTLoader": [".rst"],
    # "UnstructuredRTFLoader": [".rtf"],
    # "SRTLoader": [".srt"],
    # "TomlLoader": [".toml"],
    # "UnstructuredTSVLoader": [".tsv"],
    "UnstructuredWordDocumentLoader": [".docx"],
    # "UnstructuredXMLLoader": [".xml"],
    "UnstructuredPowerPointLoader": [".ppt", ".pptx"],
    # "EverNoteLoader": [".enex"],
}
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]

def get_LoaderClass(file_extension):
    """
    Loader获取器,根据文件类型加载文档解析器
    """
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass
        
def get_loader(loader_name: str, file_path: str, loader_kwargs: Dict = None) -> UnstructuredFileLoader:
    """
    根据loader_name和文件路径或内容返回文档加载器。
    """
    loader_kwargs = loader_kwargs or {}
    try:
        # TODO 这里方法需要重命名
        if loader_name in [
            "PDFLoader",
            "OCRLoader",
            "CSVLoader",
            "DocLoader",
            "PPTLoader",
        ]:
            document_loaders_module = importlib.import_module("app.rag.file_loader")
        else:
            document_loaders_module = importlib.import_module(
                "langchain_community.document_loaders"
            )
        DocumentLoader = getattr(document_loaders_module, loader_name)

    except Exception as e:
        msg = f"文件{file_path}查找加载器{loader_name}时出错：{e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        document_loaders_module = importlib.import_module(
            "langchain_community.document_loaders"
        )
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

    if loader_name == "UnstructuredFileLoader":
        loader_kwargs.setdefault("autodetect_encoding", True)

    elif loader_name == "CSVLoader":
        if not loader_kwargs.get("encoding"):
            # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file_path, "rb") as struct_file:
                encode_detect = chardet.detect(struct_file.read())
            if encode_detect is None:
                encode_detect = {"encoding": "utf-8"}
            loader_kwargs["encoding"] = encode_detect["encoding"]

    loader: UnstructuredFileLoader = DocumentLoader(file_path, **loader_kwargs)
    return loader

def get_kb_path(knowledge_base_name: str):
    return os.path.join(config.kb_path, knowledge_base_name)

def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_file_path(knowledge_base_name: str, doc_name: str):
    doc_path = Path(get_doc_path(knowledge_base_name)).resolve()
    file_path = (doc_path / doc_name).resolve()
    if str(file_path).startswith(str(doc_path)):
        return str(file_path)

class KnowledgeFile:
    def __init__(
        self,
        filename: str,
        knowledge_base_name: str,
        loader_kwargs: Dict = {},
    ):
        """
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。
        """
        self.kb_name = knowledge_base_name
        self.filename = str(Path(filename).as_posix())
        self.ext = os.path.splitext(filename)[-1].lower()
        if self.ext not in SUPPORTED_EXTS:
            raise ValueError(f"暂未支持的文件格式 {self.filename}")
        self.loader_kwargs = loader_kwargs
        self.filepath = get_file_path(knowledge_base_name, filename)
        self.docs = None
        self.splited_docs: Optional[List[Document]] = None
        self.document_loader_name = get_LoaderClass(self.ext)
        self.text_splitter_name = "ChineseRecursiveTextSplitter"

    def file2docs(self, refresh: bool = False):
        if self.docs is None or refresh:
            logger.info(f"使用{self.document_loader_name}解析：{self.filepath}中")
            loader = get_loader(
                loader_name=self.document_loader_name,
                file_path=self.filepath,
                loader_kwargs=self.loader_kwargs,)
            # TODO 暂不支持 md文件格式， 后续可以考虑支持
            # if isinstance(loader, TextLoader):
            #     loader.encoding = "utf8"
            #     self.docs = loader.load()
            # else:
            #     self.docs = loader.load()
            self.docs = loader.load()
        return self.docs

    def docs2texts(
        self,
        docs: List[Document] = None,
        zh_title_enhance: bool = True,
        refresh: bool = False,
        chunk_size: int = 4000,
        chunk_overlap: int = 800,
        text_splitter: TextSplitter = None,
    ) -> list | List[Document]:
        docs = docs or self.file2docs(refresh=refresh)
        if not docs:
            return []
        if self.ext not in [".csv"]:
            if text_splitter is None:
                text_splitter = make_text_splitter(
                    splitter_name=self.text_splitter_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            if self.text_splitter_name == "MarkdownHeaderTextSplitter":
                docs = text_splitter.split_text(docs[0].page_content)
            else:
                docs = text_splitter.split_documents(docs)

        if not docs:
            return []

        logger.info(f"文档切分示例：{docs[0]}")
        if zh_title_enhance:
            docs = func_zh_title_enhance(docs)
        self.splited_docs = docs
        return self.splited_docs

    def file2text(
        self,
        zh_title_enhance: bool = False,
        refresh: bool = False,
        # FIXME 此处的zh_title_enhance， chunk_size， chunk_overlap都需要改变为常量
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        text_splitter: TextSplitter = None,
    ) -> list | List[Document]:
        """_summary_

        Args:
            zh_title_enhance (bool, optional): _description_. Defaults to False.
            refresh (bool, optional): _description_. Defaults to False.
            chunk_size (int, optional): _description_. Defaults to 4000.
            chunk_overlap (int, optional): _description_. Defaults to 200.
            text_splitter (TextSplitter, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if self.splited_docs is None or refresh:
            docs = self.file2docs()
            self.splited_docs = self.docs2texts(
                docs=docs,
                zh_title_enhance=zh_title_enhance,
                refresh=refresh,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
        return self.splited_docs

    def file_exist(self):
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        return os.path.getmtime(self.filepath)

    def get_size(self):
        return os.path.getsize(self.filepath)
    

@lru_cache()
def make_text_splitter(splitter_name, chunk_size, chunk_overlap):
    """根据参数获取分词器

    Args:
        splitter_name (_type_): 分词器名称
        chunk_size (_type_): 切片大小
        chunk_overlap (_type_): 重叠大小

    Returns:
        _type_: _description_
    """
    splitter_name = splitter_name or "SpacyTextSplitter"
    text_splitter = None
    try:
        if splitter_name:  # 优先使用用户自定义的text_splitter
            text_splitter_module = importlib.import_module("app.rag.file_splitter")
            TextSplitter = getattr(text_splitter_module, splitter_name)
            text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:  # 否则使用langchain的text_splitter
            text_splitter_module = importlib.import_module("langchain.text_splitter")
            TextSplitter = getattr(text_splitter_module, splitter_name)
    except Exception as e:
        logger.exception(f"发生错误{traceback.format_exc()}")
        text_splitter_module = importlib.import_module("langchain.text_splitter")
        TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
        text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter


def files2docs_in_thread_excutor(*, file: KnowledgeFile, **kwargs) -> Tuple[bool, Tuple[str, str, List[Document]]]:
    """封装上传的文档及其参数

    Args:
        file (KnowledgeFile): _description_

    Returns:
        Tuple[bool, Tuple[str, str, List[Document]]]: _description_
    """
    try:
        return True, (file.kb_name, file.filename, file.file2text(**kwargs))
    except Exception as e:
        msg = f"从文件 {file.kb_name}/{file.filename} 加载文档时出错：{e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return False, (file.kb_name, file.filename, msg)


def files2docs_in_thread(
    files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    zh_title_enhance: bool = True,
) -> Generator:
    """
    利用多线程批量将磁盘文件转化成langchain Document.
    如果传入参数是Tuple，形式为(filename, kb_name)
    生成器返回值为 status, (kb_name, file_name, docs | error)
    """

    kwargs_list = []
    for _, file in enumerate(files):
        kwargs = {}
        try:
            if isinstance(file, tuple) and len(file) >= 2:
                filename = file[0]
                kb_name = file[1]
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            elif isinstance(file, dict):
                filename = file.pop("filename")
                kb_name = file.pop("kb_name")
                kwargs.update(file)
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            kwargs["file"] = file
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
            kwargs["zh_title_enhance"] = zh_title_enhance
            kwargs_list.append(kwargs)
        except Exception as e:
            yield False, (kb_name, filename, str(e))

    for result in thread_pool_executor(func=files2docs_in_thread_excutor, params=kwargs_list):
        yield result


# def format_reference(kb_name: str, docs: List[Dict], api_base_url: str="") -> List[Dict]:
#     '''
#     将知识库检索结果格式化为参考文档的格式
#     '''
#     from chatchat.server.utils import api_address
#     api_base_url = api_base_url or api_address(is_public=True)

#     source_documents = []
#     for inum, doc in enumerate(docs):
#         filename = doc.get("metadata", {}).get("source")
#         parameters = urlencode(
#             {
#                 "knowledge_base_name": kb_name,
#                 "file_name": filename,
#             }
#         )
#         api_base_url = api_base_url.strip(" /")
#         url = (
#             f"{api_base_url}/knowledge_base/download_doc?" + parameters
#         )
#         page_content = doc.get("page_content")
#         ref = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{page_content}\n\n"""
#         source_documents.append(ref)
    
#     return source_documents