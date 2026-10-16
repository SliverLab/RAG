from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import os
import re
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
def load_chunks(file_path):
    with open(file_path,"r",encoding="utf-8")as f:
        content=f.read()
    chunks=[]
    raw_chunks=re.split(r"(?=【)",content)
    for sub_chunks in raw_chunks:
        sub_chunks=sub_chunks.strip()#每一大段
        sub_chunk=re.split(r"(?=\n\s\n)",sub_chunks)
        for sub in sub_chunk:
            if(len(sub)>=10):
                sub=sub.strip()
                chunks.append(sub)
    return chunks
file_path=r"D:\计算机\agent\Rag\product_manual.txt"
if not os.path.exists(file_path):
    print(f"{file_path}文件不存在，请检查")
    exit(0)
chunks=load_chunks(file_path)
chunks_vector=[model.encode(code).tolist()for code in chunks]
data=[]
for i,chunk in enumerate(chunks):
    data.append(
        {
            "id":i,
            "vector":chunks_vector[i],
            "text":chunk
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
        output_fields=["text"],
        limit=2,
    )
    return res
def answer_by_LLM(question):
    results=search(question)
    res=[]
    for hit in results[0]:
        text=hit["entity"]["text"]
        print(f"相似度：{hit['distance']}")
        res.append(text)
    search_results="\n".join(res)
    system_prompt="""你是一个严谨的智能音箱客服。请根据【参考资料】回答用户问题。

回答规则：
1. 如果参考资料中有明确答案，请直接引用原文回答。
2. 如果用户用词（如"保质期"）与手册用词（如"保修期"）不同，请自动对应。
3. 如果参考资料中完全没有相关信息，请回答"抱歉，产品手册中未提及"。
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
    question=input("请输入您的问题")
    if question.lower() in["quit","q","exit","退出"]:
        break
    response=answer_by_LLM(question)
    for chunk in response:
        try:
            content=chunk.choices[0].delta.content or ""
        except:
            content=""
        print(content,flush=True,end="")