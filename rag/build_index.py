
import os,sys,json,math
import numpy as np, torch, faiss
from transformers import AutoTokenizer, AutoModel
base=r"E:KaggleLLMDepression_RAG_OA"
snap=os.path.join(base,"models","embeddings","bge-m3")
chunks=[json.loads(l) for l in open(os.path.join(base,"data","kb_clean","chunks.jsonl"),encoding="utf-8")]
texts=[c["text"] for c in chunks]
print("块数:", len(texts))
print("加载 bge-m3 ...")
tok=AutoTokenizer.from_pretrained(snap); model=AutoModel.from_pretrained(snap); model.eval()
def embed(ts, bs=16):
    out=[]
    for i in range(0,len(ts),bs):
        ins=tok(ts[i:i+bs],padding=True,truncation=True,max_length=512,return_tensors="pt")
        with torch.no_grad(): o=model(**ins)
        v=torch.nn.functional.normalize(o.last_hidden_state[:,0],dim=-1)
        out.append(v.cpu().numpy().astype("float32"))
    return np.vstack(out)
E=embed(texts); print("向量:", E.shape)
faiss.normalize_L2(E)
idx=faiss.IndexFlatIP(E.shape[1]); idx.add(E)
os.makedirs(os.path.join(base,"db2"),exist_ok=True)
faiss.write_index(idx, os.path.join(base,"db2","dense.faiss"))
# BM25 tokens
try:
    import jieba
    tokf=lambda s:[w for w in jieba.cut(s) if w.strip()]
    mode="jieba"
except Exception:
    tokf=lambda s:[s[i:i+2] for i in range(len(s)-1)]
    mode="char2gram"
corpus=[tokf(t) for t in texts]
json.dump({"mode":mode,"tokens":corpus}, open(os.path.join(base,"db2","bm25_tokens.json"),"w",encoding="utf-8"), ensure_ascii=False)
json.dump(chunks, open(os.path.join(base,"db2","chunks.json"),"w",encoding="utf-8"), ensure_ascii=False)
print("分词:", mode, "| 已写 db2/ (dense.faiss, bm25_tokens.json, chunks.json)")
# 冒烟测试
q="吃抗抑郁药会上瘾吗？"
qi=tok([q],return_tensors="pt",padding=True,truncation=True,max_length=512)
with torch.no_grad(): qv=model(**qi).last_hidden_state[:,0]
qv=torch.nn.functional.normalize(qv,dim=-1).cpu().numpy().astype("float32"); faiss.normalize_L2(qv)
sc,ix=idx.search(qv,5)
print("\n检索冒烟:", q)
for s,i in zip(sc[0],ix[0]):
    c=chunks[i]; print("  %.3f | %s p%s | %s" % (float(s), c["source"][:22], c["page"], c["text"][:70].replace(chr(10)," ")))
