from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import os
import re
from typing import Dict,List
env_path=r"D:\计算机\agent\.env"
load_dotenv(dotenv_path=env_path)
MilvClient=MilvusClient(
    uri=os.getenv("ZILLIZ_ENDPOINT"),
    token=os.getenv("ZILLIZ_TOKEN"),
)
ChatClient=OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)
model=SentenceTransformer(r"D:\计算机\agent\Rag\model_zh")

#按照大的层次去分
def  _split_paragraphs_with_headings(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path}文件路径不存在，请检查")
        exit(0)
    with open(file_path,"r",encoding="utf-8")as f:
        text=f.read()
    lines=text.splitlines()
    heading_stack:List[str]=[]
    buf:List[str]=[]
    paragraphs:List[Dict]=[]
    char_pos=0
    def flush_buf(char_pos):
        if not buf:
            return
        content="\n".join(buf)
        if content=="":
            return
        paragraphs.append(
            {
                "content":content,
                "heading_path":">".join(heading_stack),
                "start":max(0,char_pos-len(content)),
                "end":char_pos
            }
        )
    for ln in lines:
        raw=ln
        if raw.strip().startswith("#"):
            flush_buf(char_pos)
            buf=[]
            raw=raw.strip()
            title=raw.lstrip("#").strip()
            level=len(raw)-len(raw.lstrip("#"))
            if(level<=0):
                level=1
            if(level<=len(heading_stack)):
                heading_stack=heading_stack[:level-1]
            heading_stack.append(title)
            char_pos+=len(raw)+1
            continue
        if raw.strip()=="":
            flush_buf(char_pos)
            buf=[]
        else:
            buf.append(raw)
        char_pos+=len(raw)+1
    flush_buf(char_pos)
    if not paragraphs:
        paragraphs=[{"content":text,"heading_path":None,"start":0,"end":len(text)}]
    return paragraphs

#paragraphs的格式[{},{},{}]
def _approx_token_len(text: str) -> int:
    cjk = sum(1 for ch in text if _is_cjk(ch))
    count = 0
    for t in text.split():
        # 如果这个分片里面没有任何汉字，才算做non‑cjk token
        has_cjk = any(_is_cjk(ch) for ch in t)#any(生成器)：只要迭代器里面任意一个结果是True，整体返回True；全部False才返回False
        if t and not has_cjk:
            count +=1
    return cjk + count


def _is_cjk(ch: str) -> bool:#只识别汉字，标点，英文逗号句号，中文全角标点不算CJK
    """判断是否为CJK字符"""
    code = ord(ch)#ord在python中用于获取字符unicode区间
    return (
        0x4E00 <= code <= 0x9FFF or  # CJK统一汉字，日常使用汉字
        0x3400 <= code <= 0x4DBF or  # CJK扩展A，生僻字
        0x20000 <= code <= 0x2A6DF or # CJK扩展B，超大生僻古汉字，甲骨文等
        0x2A700 <= code <= 0x2B73F or # CJK扩展C#古文献，方言生僻字
        0x2B740 <= code <= 0x2B81F or # CJK扩展D#急需增补的生僻字
        0x2B820 <= code <= 0x2CEAF or # CJK扩展E#古籍，方言生僻字
        0xF900 <= code <= 0xFAFF      # CJK兼容汉字#重复汉字，用于就编码转换兼容
    )
def _chunk_paragraphs(paragraphs:List[Dict],chunks_tokens:int,overlap_tokens:int)->List[Dict]:
    chunks:List[Dict]=[]
    cur:List[Dict]=[]
    cur_tokens=0
    i=0
    while i<len(paragraphs):
        print(f"处理段落 {i}/{len(paragraphs)}，当前块tokens: {cur_tokens}")
        p=paragraphs[i]
        p_tokens=_approx_token_len(p["content"])or 1
        if p_tokens+cur_tokens<=chunks_tokens or not cur:
            cur.append(p)
            i+=1
        else:
            content="\n\n".join(x["content"] for x in cur)
            start=cur[0]["start"]
            end=cur[-1]["end"]
            heading_path=next((x["heading_path"]for x in reversed(cur) if x["heading_path"]),None)
            chunks.append(
                {
                    "content":content,
                    "start":start,
                    "end":end,
                    "heading_path":heading_path
                }
            )
            if overlap_tokens>0 and cur:
                kept:List[Dict]=[]
                kept_tokens=0
                for x in reversed(cur):
                    t=_approx_token_len(x["content"])or 1
                    if t+kept_tokens>overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens+=t
                cur=list(reversed(kept))
                cur_tokens=kept_tokens
                if cur_tokens+p_tokens>chunks_tokens:
                    cur=[]
                    cur_tokens=0
            else:
                cur_tokens=0
                cur=[]
    if cur:
        content="\n\n".join(x["content"]for x in cur)
        start=cur[0]["start"]
        end=cur[-1]["end"]
        heading_path=next((x["heading_path"]for x in reversed(cur) if x["heading_path"]),None)
        chunks.append(
            {
                "content":content,
                "start":start,
                "end":end,
                "heading_path":heading_path
            }
        )
    return chunks


file_path=r"D:\个人文件\桌面\作业\教学材料\个人日常记录文档.md"
if not os.path.exists(file_path):
    print(f"{file_path}文件不存在，请检查")
    exit(0)
paragraphs=_split_paragraphs_with_headings(file_path)
chunks_paragraphs=_chunk_paragraphs(paragraphs,50,10)#这个是没有那么碎的paragraphs
chunks_content=[]
for paragraph in chunks_paragraphs:
    chunks_content.append(paragraph.get("content"))
title=[p.get("heading_path")for p in chunks_paragraphs]
print("开始向量化，请稍候...")
chunks_vector=[model.encode(code).tolist()for code in chunks_content]
print("向量化完成！")
data=[]
for i,chunk in enumerate(chunks_content):
    data.append(
        {
            "id":i,
            "vector":chunks_vector[i],
            "text":chunk,
            "title":title[i]
        }
    )
collection_name="rag_test"
if MilvClient.has_collection(collection_name=collection_name):
    MilvClient.drop_collection(collection_name=collection_name)
    print("相关表单已经清除完毕")
MilvClient.create_collection(
    collection_name=collection_name,
    dimension=768,
)
MilvClient.insert(
    collection_name=collection_name,
    data=data,
)



#检索

MilvusClient(
    uri=os.getenv("ZILLIZ_ENDPOINT"),
    token=os.getenv("ZILLIZ_TOKEN"),
)


def search(question):
    ques_vector=[model.encode(question).tolist()]
    res=MilvClient.search(
        collection_name=collection_name,
        data=ques_vector,
        output_fields=["text","title"],
        limit=2,
    )
    return res
def answer_by_LLM(question):
    results=search(question)
    res=[]
    for hit in results[0]:
        text=hit["entity"]["text"]
        title=hit["entity"]["title"]or "无标题"
        print(f"相似度：{hit['distance']}")
        print(f"标题：{title}")
        print(f"内容：{text}")
        res.append(text)
    search_results="\n".join(res)
    system_prompt = """你是一个严谨的客服。请根据【参考资料】回答用户问题。

回答规则：
1. 如果参考资料中有明确答案，请直接引用原文回答。
2. 如果用户用词与参考资料中的表述不同但意思一致，请自动对应。
3. 如果参考资料中完全没有相关信息，请回答"抱歉，参考资料中未提及"。
4. 不要编造任何信息。"""
    user_prompt=f"【参考资料】：{search_results},用户问题：{question}"
    message=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt},
    ]
    response=ChatClient.chat.completions.create(
        model=os.getenv("LLM_MODEL_NAME"),
        messages=message,
        temperature=0.1,
        stream=True
    )
    return response
while True:
    question=input("请输入您的问题：")
    if question.lower() in["quit","q","exit","退出"]:
        break
    response=answer_by_LLM(question)
    for chunk in response:
        try:
            content=chunk.choices[0].delta.content or ""
        except:
            content=""
        print(content,flush=True,end="")
    print()