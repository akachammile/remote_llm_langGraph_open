import re
from app.logger import logger
from typing import Union, List
from langchain.docstore.document import Document
from app.tools.custom_decorator_tool import retry_decorator


def under_non_alpha_ratio(text: str, threshold: float = 0.5) -> bool:
    """检查文本片段中非字母字符的比例是否超过给定阈值。  
       这有助于防止像 "-----------BREAK---------" 这样的文本被标记为标题或叙述文本。  
       比例计算时不包括空格。

        Args:
            text(str): 要测试的输入字符串
            threshold(float): 如果非字母字符的比例超过此阈值，函数将返回 False
    """
    if len(text) == 0:
        return False

    alpha_count = len([char for char in text if char.strip() and char.isalpha()])
    total_count = len([char for char in text if char.strip()])
    try:
        ratio = alpha_count / total_count
        return ratio < threshold
    except:
        return False

    # """Checks to see if the text passes all of the checks for a valid title.

    # Parameters
    # ----------
    # text
    #     The input text to check
    # title_max_word_length
    #     The maximum number of words a title can contain
    # non_alpha_threshold
    #     The minimum number of alpha characters the text needs to be considered a title
    # """

@retry_decorator
def is_possible_title(
    text: str,
    title_max_word_length: int = 20,
    non_alpha_threshold: float = 0.5,
) -> bool:
    """检查文本是否通过了所有有效标题的校验。

    Args:
        text (str): 要检查的输入文本
        title_max_word_length (int, optional):  标题所能包含的最大单词数. Defaults to 20.
        non_alpha_threshold (float, optional):  文本中所需的最少字母字符数，以使其被视为标题. Defaults to 0.5.

    Returns:
        bool: 判断是否是title
    """
    # 文本长度为0的话，肯定不是title
    if len(text) == 0:
        print("非title")
        return False

    # 文本中有标点符号，就不是title
    ENDS_IN_PUNCT_PATTERN = r"[^\w\s]\Z"
    ENDS_IN_PUNCT_RE = re.compile(ENDS_IN_PUNCT_PATTERN)
    if ENDS_IN_PUNCT_RE.search(text) is not None:
        return False

    # 文本长度不能超过设定值，默认20
    if len(text) > title_max_word_length:
        return False

    # 文本中数字的占比不能太高，否则不是title
    if under_non_alpha_ratio(text, threshold=non_alpha_threshold):
        return False

    # NOTE(robinson) - Prevent flagging salutations like "To My Dearest Friends," as titles
    if text.endswith((",", ".", "，", "。")):
        return False

    if text.isnumeric():
        (f"数字无法作为title:\n\n{text}")  # type: ignore
        return False

    # 开头的字符内应该有数字，默认5个字符内
    if len(text) < 5:
        text_5 = text
    else:
        text_5 = text[:5]
    alpha_in_text_5 = sum(list(map(lambda x: x.isnumeric(), list(text_5))))
    if not alpha_in_text_5:
        return False

    return True


def zh_title_enhance(docs: Union["Document", List["Document"]]) -> Document:
    """_summary_

    Args:
        docs (Document): 文档

    Returns:
        Document: Document对象
    """
    # 统一成列表处理
    if not isinstance(docs, list):
        docs = [docs]
    title = None

    if len(docs) > 0:
        for doc in docs:
            if is_possible_title(doc.page_content):
                doc.metadata["category"] = "cn_Title"
                title = doc.page_content
            elif title:
                doc.page_content = f"下文与({title})有关。{doc.page_content}"
        # FIXME 此处可能会有BUG，后面估计有问题
        return docs
    else:
        logger.warning("文件不存在")
        
