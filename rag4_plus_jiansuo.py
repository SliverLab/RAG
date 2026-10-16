from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import os
import re
from typing import Dict, List, Optional

env_path = r"D:\计算机\agent\.env"
load_dotenv(dotenv_path=env_path)
MilvClient = MilvusClient(
    uri=os.getenv("ZILLIZ_ENDPOINT"),
    token=os.getenv("ZILLIZ_TOKEN"),
)
ChatClient = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)
model = SentenceTransformer(r"D:\计算机\agent\Rag\model_zh")


# ==================== 文档分割 ====================
def _split_paragraphs_with_headings(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path}文件路径不存在，请检查")
        exit(0)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines()
    heading_stack: List[str] = []
    buf: List[str] = []
    paragraphs: List[Dict] = []
    char_pos = 0

    def flush_buf(char_pos):
        if not buf:
            return
        content = "\n".join(buf)
        if content == "":
            return
        paragraphs.append(
            {
                "content": content,
                "heading_path": ">".join(heading_stack),
                "start": max(0, char_pos - len(content)),
                "end": char_pos,
            }
        )

    for ln in lines:
        raw = ln
        if raw.strip().startswith("#"):
            flush_buf(char_pos)
            buf = []
            raw = raw.strip()
            title = raw.lstrip("#").strip()
            level = len(raw) - len(raw.lstrip("#"))
            if level <= 0:
                level = 1
            if level <= len(heading_stack):
                heading_stack = heading_stack[:level - 1]
            heading_stack.append(title)
            print(f"标题: {title}, 级别: {level}, 栈: {heading_stack}")
            char_pos += len(raw) + 1
            continue
        if raw.strip() == "":
            flush_buf(char_pos)
            buf = []
        else:
            buf.append(raw)
        char_pos += len(raw) + 1
    flush_buf(char_pos)
    if not paragraphs:
        paragraphs = [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]
    return paragraphs


# ==================== Token 估算 ====================
def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or
        0x3400 <= code <= 0x4DBF or
        0x20000 <= code <= 0x2A6DF or
        0x2A700 <= code <= 0x2B73F or
        0x2B740 <= code <= 0x2B81F or
        0x2B820 <= code <= 0x2CEAF or
        0xF900 <= code <= 0xFAFF
    )


def _approx_token_len(text: str) -> int:
    cjk = sum(1 for ch in text if _is_cjk(ch))
    count = 0
    for t in text.split():
        has_cjk = any(_is_cjk(ch) for ch in t)
        if t and not has_cjk:
            count += 1
    return cjk + count


# ==================== Token 分块 ====================
def _chunk_paragraphs(paragraphs: List[Dict], chunks_tokens: int, overlap_tokens: int) -> List[Dict]:
    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        p_tokens = _approx_token_len(p["content"]) or 1
        print(f"段落{i}: p_tokens={p_tokens}, cur_tokens={cur_tokens}")
        if p_tokens + cur_tokens <= chunks_tokens or not cur:
            cur.append(p)
            i += 1
            cur_tokens=p_tokens+cur_tokens
        else:
            content = "\n\n".join(x["content"] for x in cur)
            start = cur[0]["start"]
            end = cur[-1]["end"]
            heading_path = next((x["heading_path"] for x in cur if x["heading_path"]), None)  # ✅ 改为 cur
            chunks.append({"content": content, "start": start, "end": end, "heading_path": heading_path})

            if overlap_tokens > 0 and cur:
                kept: List[Dict] = []
                kept_tokens = 0
                for x in reversed(cur):
                    t = _approx_token_len(x["content"]) or 1
                    if kept_tokens + t > overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens += t
                cur = list(reversed(kept))
                cur_tokens = kept_tokens
                if cur_tokens + p_tokens > chunks_tokens:
                    cur = []
                    cur_tokens = 0
            else:
                cur = []
                cur_tokens = 0
    if cur:
        content = "\n\n".join(x["content"] for x in cur)
        start = cur[0]["start"]
        end = cur[-1]["end"]
        heading_path = next((x["heading_path"] for x in cur if x["heading_path"]), None)  # ✅ 改为 cur
        chunks.append({"content": content, "start": start, "end": end, "heading_path": heading_path})
    return chunks


# ==================== 准备文档并建库 ====================
file_path = r"D:\个人文件\桌面\作业\教学材料\个人日常记录文档.md"
if not os.path.exists(file_path):
    print(f"{file_path}文件不存在，请检查")
    exit(0)
paragraphs = _split_paragraphs_with_headings(file_path)
print(f"原始段落数: {len(paragraphs)}")
chunks_paragraphs = _chunk_paragraphs(paragraphs, 100,20)

print(ch["content"] for ch in chunks_paragraphs)
chunks_content = [p.get("content") for p in chunks_paragraphs]
title = [p.get("heading_path") for p in chunks_paragraphs]

print("开始向量化，请稍候...")
chunks_vector = [model.encode(code).tolist() for code in chunks_content]
print("向量化完成！")
print(f"分块后数量: {len(chunks_paragraphs)}")
for i, c in enumerate(chunks_paragraphs):
    print(f"块{i}: 标题={c['heading_path']} | 内容开头={c['content'][:40]}")
data = []
for i, chunk in enumerate(chunks_content):
    data.append({
        "id": i,
        "vector": chunks_vector[i],
        "text": chunk,
        "title": title[i],
    })

collection_name = "rag_test"
if MilvClient.has_collection(collection_name=collection_name):
    MilvClient.drop_collection(collection_name=collection_name)
    print("相关表单已经清除完毕")
MilvClient.create_collection(collection_name=collection_name, dimension=768)
MilvClient.insert(collection_name=collection_name, data=data)


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


# ==================== 问答 ====================
def answer_by_LLM(question):
    results = search_vectors_expanded(
        store=MilvClient,
        query=question,
        collection_name=collection_name,
        top_k=2,
        enable_mqe=True,
        mqe_expansions=2,
    )
    res = []
    for hit in results:                     # ✅ 直接遍历 results，不是 results[0]
        text = hit["entity"]["text"]
        title = hit["entity"]["title"] or "无标题"
        print(f"相似度：{hit['distance']}")
        print(f"标题：{title}")
        print(f"内容：{text}")
        res.append(text)
    search_results = "\n".join(res)

    system_prompt = """你是一个严谨的客服。请根据【参考资料】回答用户问题。

回答规则：
1. 如果参考资料中有明确答案，请直接引用原文回答。
2. 如果用户用词与参考资料中的表述不同但意思一致，请自动对应。
3. 如果参考资料中完全没有相关信息，请回答"抱歉，参考资料中未提及"。
4. 不要编造任何信息。"""
    user_prompt = f"【参考资料】：{search_results},用户问题：{question}"
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = ChatClient.chat.completions.create(
        model=os.getenv("LLM_MODEL_NAME"),
        messages=message,
        temperature=0.1,
        stream=True,
    )
    return response


# ==================== 交互 ====================
while True:
    question = input("请输入您的问题：")
    if question.lower() in ["quit", "q", "exit", "退出"]:
        break
    response = answer_by_LLM(question)
    for chunk in response:
        try:
            content = chunk.choices[0].delta.content or ""
        except:
            content = ""
        print(content, flush=True, end="")
    print()