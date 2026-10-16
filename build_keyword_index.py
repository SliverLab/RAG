import hashlib
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import re
env_path=r"D:\计算机\agent\.env"
load_dotenv(dotenv_path=env_path)
ChatClient=OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)
def build_keyword_index(chunks,cache_dir="keyword_cache"):
    text="".join(chunks)
    hash_doc=hashlib.md5(text.encode()).hexdigest()[:8]
    file_path=os.path.join(cache_dir,f"{hash_doc}.json")
    if os.path.exists(file_path):
        print(f"关键词已经缓存到{file_path}中了")
        with open(file_path,"r",encoding="utf-8")as f:
            raw=json.load(f)
            keyword_index={int(k):v for k,v in raw.items()}
            return keyword_index
    print("第一次加载切片关键词，请耐心等待")
    keyword_index={}
    model_name=os.getenv("LLM_MODEL_NAME")
    if not model_name:
        print("请配置环境变量MODLE_NAME")
        return {}
    for i,chunk in enumerate(chunks):
        if not chunk.strip():
            keyword_index[i]=[]
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
        message=[
            {"role":"system","content":prompt},
            {"role":"user","content":chunk}
        ]
        response=ChatClient.chat.completions.create(
            model=model_name,
            messages=message,
            temperature=float(os.getenv("LLM_TEMPERATURE"),"0.1"),
        )
        content_raw=response.choices[0].message.content.strip()
        keyword=[]
        try:
            json_match=re.search(r"\{[\s\S]*?}",content_raw)
            if json_match:
                results=json.loads(json_match.group())
                keyword=results["keywords"]
        except json.JSONDecodeError:
            cleaned=content_raw.replace("，",",")
            raw_list=cleaned.split(",")
            keyword=[kw.strip() for kw in raw_list if kw.strip()]
        keyword_index[i]=keyword
    os.makedirs(cache_dir,exist_ok=True)
    if keyword_index:
        with open(file_path,"w",encoding="utf-8")as f:
            json.dump(keyword_index,f,ensure_ascii=False,indent=2)
    return keyword_index