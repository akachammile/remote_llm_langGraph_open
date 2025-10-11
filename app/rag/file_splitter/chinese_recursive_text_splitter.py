import re
from app.logger import logger
from typing import Any, List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter


def _split_text_with_regex_from_end(text: str, separator: str, keep_separator: bool) -> List[str]:
    """根据标点符号分割文本

    Args:
        text (str): 文本
        separator (str): 符号
        keep_separator (bool): 是否保留符号

    Returns:
        List[str]: 分割后的List
    """
    if separator:
        if keep_separator:
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = True,
        **kwargs: Any,
    ) -> None:
        """创建中文text的Splitter

        Args:
            separators (Optional[List[str]], optional): 标点符号. Defaults to None.
            keep_separator (bool, optional): 是否保留符号. Defaults to True.
            is_separator_regex (bool, optional): 是否符号正则表达式. Defaults to True.
        """
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = [
            "\n\n",
            "\n",
            " ",
            r"\u3002\s|\.\s|\!\s|\?\s",
            r"；|;\s",
            r"，|,\s",
            "",
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """内部私有类， 文本切片

        Args:
            text (str): 文本
            separators (List[str]): 符号

        Returns:
            List[str]: 分割后的文本List
        """
        final_chunks: List[str] = []

        # 获取合适的分割符号
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # 如果文本很长，迭代处理
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [
            re.sub(r"\n{2,}", "\n", chunk.strip())
            for chunk in final_chunks
            if chunk.strip() != ""
        ]