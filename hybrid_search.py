from openai import OpenAI
from dotenv import load_dotenv
import json
import re
import os
from typing import Dict,List,Optional
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
model=SentenceTransformer(r"D:\计算机\agent\Rag\model_zh")
env_path=r"D:\计算机\agent\.env"
load_dotenv(dotenv_path=env_path)
ChatClient=OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

MilvClient=MilvusClient(
    uri=os.getenv("ZILLIZ_ENDPOINT"),
    token=os.getenv("ZILLIZ_TOKEN"),
)

def _extract_query_keywords(question:str)->list:
    prompt='''你是专业文本关键词提取器。
任务：从用户输入的文本中提取核心关键词。
规则：
1. 只输出标准JSON数组，不要解释、不要多余文字、不要markdown标题。
2. 关键词为名词、名词短语、核心实体、关键概念。
3. 去重，不输出语义重复词汇。
4. 禁止输出任何其他内容，严格示例输出：["关键词1","关键词2","关键词3"]'''
    message=[
        {"role":"system","content":prompt},
        {"role":"user","content":question},
    ]
    model_name=os.getenv("LLM_MODEL_NAME")
    if not model_name:
        print("请配置环境变量LLM_MODEL_NAME")
        return []
    temp_str=os.getenv("LLM_TEMPERATURE")
    temperature=float(temp_str) if temp_str is not None else 0.1
    response=ChatClient.chat.completions.create(
        model=model_name,
        messages=message,
        temperature=temperature
    )
    content=response.choices[0].message.content.strip()
    if not content:
        print("大模型输出为空")
        return []
    try:
        keyword_list=json.loads(content)
        if not isinstance(keyword_list,list):
            return []
        keyword_list=[kw.strip() for kw in keyword_list if kw.strip()]
        return keyword_list
    except json.JSONDecodeError:
        print(f"JSON解析失败，模型原始输出为{content}")
        return []




# ==================== 查询扩展 ====================
def _prompt_mqe(query: str, expansions: int) -> List[str]:
    prompt = f"""你是查询改写助手。针对用户问题，生成{expansions}条语义等价、表述角度不同的搜索查询，用于知识库检索。
要求：
1. 每条简短，适合向量库检索；
2. 不要解释，不要编号，每条单独一行输出；
3. 不要返回原始问题；

用户问题：{query}
"""
    try:
        resp = ChatClient.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        raw_text = resp.choices[0].message.content.strip()
        candidates = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return candidates[:expansions]
    except Exception as e:
        print(f"MQE生成失败: {e}")
        return []


def _prompt_hyde(query: str) -> Optional[str]:
    prompt = f"""请写一段简短的参考文档片段，用来直接回答下面用户的问题。
只输出文档内容，不要标题，不要开场白，不要反问，不要解释。

用户问题：{query}
"""
    try:
        resp = ChatClient.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        hyde_doc = resp.choices[0].message.content.strip()
        return hyde_doc if hyde_doc else None
    except Exception as e:
        print(f"HyDE生成失败: {e}")
        return None


def embed_query(text: str):
    return model.encode(text).tolist()   # ✅ 返回一维列表


# ==================== 扩展检索 ====================
def search_vectors_expanded(
    store=None,
    query: str = "",
    collection_name: str = "rag_test",
    top_k: int = 3,
    enable_mqe: bool = False,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    candidate_pool_multiplier: int = 4,
) -> List[Dict]:
    if not query:
        return []
    if store is None:
        store = MilvClient

    expansions = [query]
    if enable_mqe and mqe_expansions > 0:
        expansions.extend(_prompt_mqe(query, mqe_expansions))
    if enable_hyde:
        hyde_text = _prompt_hyde(query)
        if hyde_text:
            expansions.append(hyde_text)

    unique = []
    for e in expansions:
        if e not in unique:
            unique.append(e)
    expansions = unique

    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // len(expansions))

    agg = {}
    for q in expansions:
        qv = embed_query(q)
        hits = store.search(
            collection_name=collection_name,
            data=[qv],
            limit=per,
            output_fields=["text", "title"],
        )
        for h in hits[0]:
            cid = h.get("id")
            dist = h.get("distance", 0.0)
            if cid not in agg or dist > agg[cid].get("distance", 0.0):
                agg[cid] = h

    merged = list(agg.values())
    merged.sort(key=lambda x: x.get("distance", 0.0), reverse=True)
    return merged[:top_k]

#build_keyword_index(chunks)是为每个片段单独提取关键词，返回{块id:[关键词]}
def hybrid_search(question:str,chunks,keyword_index:Dict,top_k=3):
    q_keywords=_extract_query_keywords(question)
    #is None与not q_keywords的区别：
    #is None：是空才执行，遇见空列表是不会执行的
    #not q_keywords遇见空列表，None，空字符串都会执行
    if not q_keywords:
        return[]
    hits=[]
    for chunk_id ,kws in keyword_index.items():
        if not isinstance(kws,list):
            continue
        if any(qk[:2] in kw[:2] or kw[:2] in qk[:2] for qk in q_keywords for kw in kws):
            hits.append(chunks[int(chunk_id)])
    if hits:
        return hits[:top_k]
    semantic_results= search_vectors_expanded(query=question,enable_mqe=True)
    return [hit["entity"]["text"]for hit in semantic_results]