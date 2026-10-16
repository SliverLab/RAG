# -*- coding: utf-8 -*-
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# 加载环境变量
env_path = r"D:\计算机\agent\.env"
load_dotenv(dotenv_path=env_path)

# 初始化客户端
MilvClient = MilvusClient(
    uri=os.getenv("ZILLIZ_ENDPOINT"),
    token=os.getenv("ZILLIZ_TOKEN"),
)
ChatClient = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)
model = SentenceTransformer(r"D:\计算机\agent\Rag\model_zh")


# ------------------------- 文档分割 -------------------------
def _split_paragraphs_with_headings(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        print(f"{file_path} 文件不存在")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines()
    heading_stack: List[str] = []
    buf: List[str] = []
    paragraphs: List[Dict] = []
    char_pos = 0

    def flush_buf(end_pos: int):
        if not buf:
            return
        content = "\n".join(buf).strip()
        if not content:
            return
        paragraphs.append({
            "content": content,
            "heading_path": " > ".join(heading_stack) if heading_stack else None,
            "start": max(0, end_pos - len(content)),
            "end": end_pos,
        })

    for ln in lines:
        raw = ln
        if raw.strip().startswith("#"):
            flush_buf(char_pos)
            buf = []
            raw = raw.strip()
            title = raw.lstrip('#').strip()
            level = len(raw) - len(raw.lstrip('#'))
            if level <= 0:
                level = 1
            if level <= len(heading_stack):
                heading_stack = heading_stack[:level - 1]
            heading_stack.append(title)
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


# ------------------------- Token 估算 -------------------------
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


# ------------------------- 基于 Token 的分块 -------------------------
def _chunk_paragraphs(paragraphs: List[Dict], chunk_tokens: int, overlap_tokens: int) -> List[Dict]:
    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        p_tokens = _approx_token_len(p["content"]) or 1
        if p_tokens + cur_tokens <= chunk_tokens or not cur:
            cur.append(p)
            i += 1
        else:
            content = "\n\n".join(x["content"] for x in cur)
            start = cur[0]["start"]
            end = cur[-1]["end"]
            # 修正：取当前块中第一个有标题的段落标题，避免标题错位
            heading_path = next((x["heading_path"] for x in cur if x.get("heading_path")), None)
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
                if cur_tokens + p_tokens > chunk_tokens:
                    cur = []
                    cur_tokens = 0
            else:
                cur = []
                cur_tokens = 0
    if cur:
        content = "\n\n".join(x["content"] for x in cur)
        start = cur[0]["start"]
        end = cur[-1]["end"]
        heading_path = next((x["heading_path"] for x in cur if x.get("heading_path")), None)
        chunks.append({"content": content, "start": start, "end": end, "heading_path": heading_path})
    return chunks


# ------------------------- 查询扩展 -------------------------
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
    return model.encode(text).tolist()


# ------------------------- 扩展检索 -------------------------
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
            output_fields=["text", "title"]
        )
        for h in hits[0]:
            cid = h.get("id")
            dist = h.get("distance", 0.0)
            if cid not in agg or dist > agg[cid].get("distance", 0.0):
                agg[cid] = h

    merged = list(agg.values())
    merged.sort(key=lambda x: x.get("distance", 0.0), reverse=True)
    return merged[:top_k]


# ------------------------- 主流程：建库并测试 -------------------------
if __name__ == "__main__":
    # 1. 准备文档（请确保你的文件已存在，并且标题使用 # 格式）
    file_path = r"D:\个人文件\桌面\作业\教学材料\个人日常记录文档.md"
    paragraphs = _split_paragraphs_with_headings(file_path)
    print(f"分割出 {len(paragraphs)} 个段落")
    
    # 2. Token分块
    chunks = _chunk_paragraphs(paragraphs, chunk_tokens=100, overlap_tokens=20)
    print(f"分块后得到 {len(chunks)} 个块")
    for i, c in enumerate(chunks):
        print(f"块{i}: 标题={c.get('heading_path')}, 内容前30字={c['content'][:30]}")

    # 3. 向量化并入库
    collection_name = "rag_test"
    if MilvClient.has_collection(collection_name):
        MilvClient.drop_collection(collection_name)
    MilvClient.create_collection(collection_name=collection_name, dimension=768)

    data = []
    for i, c in enumerate(chunks):
        vec = model.encode(c["content"]).tolist()
        data.append({
            "id": i,
            "vector": vec,
            "text": c["content"],
            "title": c.get("heading_path") or "无标题",
        })
    MilvClient.insert(collection_name=collection_name, data=data)
    print("数据入库完成\n")

    # 4. 测试检索
    test_queries = [
        "创建时间",
        "今日核心内容",
        "待办计划清单",
    ]
    for q in test_queries:
        print("=" * 60)
        print(f"测试查询: {q}")
        print("--- 普通检索 ---")
        results = search_vectors_expanded(store=MilvClient, query=q, collection_name=collection_name, top_k=2)
        for hit in results:
            print(f"标题: {hit['entity'].get('title')}")
            print(f"内容: {hit['entity'].get('text')[:100]}...")
            print(f"相似度: {hit['distance']:.4f}")

        print("\n--- MQE增强检索 ---")
        results_mqe = search_vectors_expanded(
            store=MilvClient, query=q, collection_name=collection_name,
            top_k=2, enable_mqe=True, mqe_expansions=2
        )
        for hit in results_mqe:
            print(f"标题: {hit['entity'].get('title')}")
            print(f"内容: {hit['entity'].get('text')[:100]}...")
            print(f"相似度: {hit['distance']:.4f}")
        print()