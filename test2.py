# -*- coding: utf-8 -*-
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import json
import re
import hashlib

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

    in_code_block = False  # ✅ 新增：跳过代码块

    for ln in lines:
        raw = ln
        if raw.strip().startswith("```"):
            in_code_block = not in_code_block
            char_pos += len(raw) + 1
            continue
        if in_code_block:
            char_pos += len(raw) + 1
            continue

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
            cur_tokens += p_tokens
            i += 1
        else:
            content = "\n\n".join(x["content"] for x in cur)
            start = cur[0]["start"]
            end = cur[-1]["end"]
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


def build_keyword_index(chunks, cache_dir="keyword_cache"):
    text = "".join(chunks)
    hash_doc = hashlib.md5(text.encode()).hexdigest()[:8]
    file_path = os.path.join(cache_dir, f"{hash_doc}.json")
    if os.path.exists(file_path):
        print(f"关键词已经缓存到{file_path}中了")
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            keyword_index = {int(k): v for k, v in raw.items()}
            return keyword_index
    print("第一次加载切片关键词，请耐心等待")
    keyword_index = {}
    model_name = os.getenv("LLM_MODEL_NAME")
    if not model_name:
        print("请配置环境变量LLM_MODEL_NAME")
        return {}
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            keyword_index[i] = []
            continue

        prompt = f"""你是一个关键词提取专家。请从以下文本片段中，提取 3-5 个最重要的关键词。

规则：
1. 关键词必须是文本中出现的词或短语，不要自己编造。
2. 优先提取：产品名称、功能、数字参数、专有名词。
3. 严格按 JSON 格式返回，不要任何解释。

文本片段：
{chunk[:300]}

返回格式：
{{"keywords": ["关键词1", "关键词2", "关键词3"]}}"""
        message = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": chunk}
        ]
        response = ChatClient.chat.completions.create(
            model=model_name,
            messages=message,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        )
        content_raw = response.choices[0].message.content.strip()
        keyword = []
        try:
            json_match = re.search(r"\{[\s\S]*?\}", content_raw)
            if json_match:
                results = json.loads(json_match.group())
                keyword = results.get("keywords", [])
        except json.JSONDecodeError:
            cleaned = content_raw.replace("，", ",")
            raw_list = cleaned.split(",")
            keyword = [kw.strip() for kw in raw_list if kw.strip()]
        keyword_index[i] = keyword
    os.makedirs(cache_dir, exist_ok=True)
    if keyword_index:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(keyword_index, f, ensure_ascii=False, indent=2)
    return keyword_index


def _extract_query_keywords(question: str) -> list:
    prompt = '''你是专业文本关键词提取器。
任务：从用户输入的文本中提取核心关键词。
规则：
1. 只输出标准JSON数组，不要解释、不要多余文字、不要markdown标题。
2. 关键词为名词、名词短语、核心实体、关键概念。
3. 去重，不输出语义重复词汇。
4. 禁止输出任何其他内容，严格示例输出：["关键词1","关键词2","关键词3"]'''
    message = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]
    model_name = os.getenv("LLM_MODEL_NAME")
    if not model_name:
        print("请配置环境变量LLM_MODEL_NAME")
        return []
    temp_str = os.getenv("LLM_TEMPERATURE")
    temperature = float(temp_str) if temp_str is not None else 0.1
    response = ChatClient.chat.completions.create(
        model=model_name,
        messages=message,
        temperature=temperature
    )
    content = response.choices[0].message.content.strip()
    if not content:
        print("大模型输出为空")
        return []
    try:
        keyword_list = json.loads(content)
        if not isinstance(keyword_list, list):
            return []
        keyword_list = [kw.strip() for kw in keyword_list if kw.strip()]
        return keyword_list
    except json.JSONDecodeError:
        print(f"JSON解析失败，模型原始输出为{content}")
        return []


def hybrid_search(question: str, chunks, keyword_index: Dict, top_k=3):
    q_keywords = _extract_query_keywords(question)
    print(f"问题关键词: {q_keywords}")
    print(f"关键词索引: {keyword_index}")
    if not q_keywords:
        return []
    hits = []
    for chunk_id, kws in keyword_index.items():
        if not isinstance(kws, list):
            continue
        # ✅ 放宽匹配：包含匹配，不再只比较前两字
        if any((qk in kw or kw in qk) for qk in q_keywords for kw in kws):
            hits.append(chunks[int(chunk_id)])
    if hits:
        return hits[:top_k]
    semantic_results = search_vectors_expanded(query=question, enable_mqe=True)
    return [hit["entity"]["text"] for hit in semantic_results]


# ------------------------- 主流程：建库并测试 -------------------------
if __name__ == "__main__":
    file_path = r"D:\个人文件\桌面\作业\教学材料\个人日常记录文档.md"
    paragraphs = _split_paragraphs_with_headings(file_path)
    print(f"分割出 {len(paragraphs)} 个段落")

    chunks = _chunk_paragraphs(paragraphs, chunk_tokens=100, overlap_tokens=20)
    print(f"分块后得到 {len(chunks)} 个块")
    for i, c in enumerate(chunks):
        print(f"块{i}: 标题={c.get('heading_path')}, 内容前30字={c['content'][:30]}")

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

    # 构建关键词索引：合并标题+内容
    chunks_for_index = []
    for c in chunks:
        title = c.get("heading_path") or ""
        content = c.get("content") or ""
        chunks_for_index.append(f"{title}\n{content}")

    keyword_index = build_keyword_index(chunks_for_index)

    # ✅ 手动补充标题词到索引，防止大模型漏掉
    for i, c in enumerate(chunks):
        heading = c.get("heading_path") or ""
        if heading:
            last_title = heading.split(">")[-1].strip()
            if last_title and last_title not in keyword_index.get(i, []):
                keyword_index.setdefault(i, []).append(last_title)

    test_queries = ["重点知识点梳理", "待办计划清单", "学习内容记录", "创建时间"]

    for q in test_queries:
        print("=" * 60)
        print(f"测试问题: {q}")
        results = hybrid_search(q, [c["content"] for c in chunks], keyword_index, top_k=2)

        if not results:
            print("没有找到结果")
            continue

        for idx, text in enumerate(results):
            print(f"结果 {idx+1}: {text[:100]}...")
        print()