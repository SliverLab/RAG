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
file_path=r"D:\个人文件\桌面\作业\教学材料\个人日常记录文档.md"
if not os.path.exists(file_path):
    print(f"{file_path}文件不存在，请检查")
    exit(0)
paragraphs=_split_paragraphs_with_headings(file_path)
chunks=[]
for paragraph in paragraphs:
    chunks.append(paragraph.get("content"))
title=[]
for paragraph in paragraphs:
    title.append(paragraph.get("heading_path"))
chunks_vector=[model.encode(code).tolist()for code in chunks]
data=[]
for i,chunk in enumerate(chunks):
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
        title=hit["entity"]["title"]
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