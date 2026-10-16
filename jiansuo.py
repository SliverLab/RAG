from typing import Optional,Dict,List
def _create_default_vector_store():
    pass
def _prompt_mqe(query:str,expansions:int)->List[str]:
    pass
def _prompt_hyde(query:str)->Optional[str]:
    pass
def embed_query(text:str):
    pass
def search_vectors_expanded(
    store=None,
    query:str="",
    top_k:int=8,
    rag_namespace:Optional[str]=None,
    only_rag_data:bool=True,
    score_threshole:Optional[float]=None,
    enable_mqe:bool=False,
    mqe_expansions:int=2,
    enable_hyde:bool=False,
    candidata_pool_multiplier:int=4,    
)->List[Dict]:
    if not query:
        return []
    if store is None:
        store=_create_default_vector_store()
    expansions:List[str]=[query]
    if enable_mqe and mqe_expansions>0:
        expansions.extend(_prompt_mqe(query,expansions))
    if enable_hyde:
        hyde_text=_prompt_hyde(query)
        if hyde_text:
            expansions.append(hyde_text)
    unique:List[str]=[]
    for e in expansions:
        if e not in unique:
            unique.append(e)
    expansions=unique[:max(1,len(unique))]
    pool=max(candidata_pool_multiplier*top_k,20)
    peer=max(1,pool//len(expansions))
    where={"memory_type":"rag_chunk"}
    if only_rag_data:
        where["is_rag_data"]=True
        where["data_source"]="rag_pipeline"
    if rag_namespace:
        where["rag_namespace"]=rag_namespace
    agg:Dict[str,Dict]={}
    for q in expansions:
        qv=embed_query(q)
        hits=store.search_similar(
            query_vector=qv,
            limit=peer,
            score_threshole=score_threshole,
            where=where,
        )
        for h in hits:
            mid=h.get("metadata",{}).get("memory_id",h.get("id"))
            s=float(h.get("score",0.0))
            if mid not in agg or s>float(agg[mid].get("score",0.0)):
                agg[mid]=h
        merged=list(agg.values())
        merged.sort(key=lambda x :float(x.get("score",0.0)),reverse=True)
        return merged[:top_k]