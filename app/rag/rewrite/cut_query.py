# import jieba

# # 加载停用词表（只需执行一次）
# # FIXME 这里需要改写为高性能
# def load_stopwords(filepath: str) -> set[str]:
#     stopwords = set()
#     with open(filepath, "r", encoding="utf-8") as f:
#         for line in f:
#             word = line.strip()
#             if word:
#                 stopwords.add(word)
#     return stopwords

# # 初始化停用词
# STOPWORDS = load_stopwords(r"E:\1_LLM_PROJECT\remote_llm_langGraph\doc\baidu_stopwords.txt")

# def cut_query(text: str) -> list[str]:
#     """分词器，自动去掉停用词

#     Args:
#         text (str): 用户问题

#     Returns:
#         list[str]: 分词结果（已去停用词）
#     """
#     words = jieba.cut(text, cut_all=False)
#     return [w for w in words if w not in STOPWORDS and w.strip()]
