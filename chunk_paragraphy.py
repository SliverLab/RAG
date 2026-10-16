from typing import Dict,List,Optional
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