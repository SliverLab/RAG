from typing import Dict,List
import os
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
            title=raw.lstrip("#")
            level=len(raw)-len(title)
            if(level<len(heading_stack)):
                heading_stack=heading_stack[:level-1]
            heading_stack.append(title)
            char_pos+=len(raw)+1
            continue
        if raw.strip=="":
            flush_buf(buf)
            buf=[]
        else:
            buf.append(raw)
        char_pos+=len(raw)+1
    flush_buf(char_pos)
    if not paragraphs:
        paragraphs=[{"content":text,"heading_path":None,"start":0,"end":len(text)}]
    return paragraphs
if __name__ =="__main__":
    file_path=r"D:\个人文件\桌面\作业\教学材料\个人日常记录文档.md"
    paragraph=_split_paragraphs_with_headings(file_path)
    for i in paragraph:
        content=i.get("content")
        title=i.get("heading_path")
        print(title)
        print(content)
        print()