# rag/engine.py  --  可信RAG引擎：检索质量 + 可溯源 + 无据拒答
import os, json, re, math
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB2 = os.path.join(BASE, "db2")
BGE_M3 = os.path.join(BASE, "models", "embeddings", "bge-m3")
RERANK_DIR = os.path.join(BASE, "models", "reranker")

def load_env():
    env = {}
    for p in [os.path.join(BASE, ".env"),
              r"F:\AgentProjects\interview-coach-agent\.env",
              r"E:\Kaggle\LLM\AgentProjects\weather-care-ai\.env"]:
        if os.path.exists(p):
            for ln in open(p, encoding="utf-8", errors="ignore"):
                ln = ln.strip()
                if ln and not ln.startswith("#") and "=" in ln:
                    k, v = ln.split("=", 1); env[k.strip()] = v.strip()
    return env
ENV = load_env()

def chat(messages, model=None, temperature=0.0, max_tokens=900):
    from openai import OpenAI
    model = model or ENV.get("RAG_MODEL", "qwen-plus")
    if model.startswith("qwen"):
        cli = OpenAI(api_key=ENV["DASHSCOPE_API_KEY"],
                     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    else:
        cli = OpenAI(api_key=ENV["DEEPSEEK_API_KEY"],
                     base_url=ENV.get("COACH_BASE_URL", "https://api.deepseek.com"))
    r = cli.chat.completions.create(model=model, messages=messages,
                                    temperature=temperature, max_tokens=max_tokens)
    return (r.choices[0].message.content or "").strip()

# ---------- 向量 ----------
_EMB = None
def _embedder():
    global _EMB
    if _EMB is None:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(BGE_M3)
        m = AutoModel.from_pretrained(BGE_M3); m.eval()
        _EMB = (tok, m)
    return _EMB

def embed(texts, bs=16):
    import torch
    tok, m = _embedder(); out = []
    for i in range(0, len(texts), bs):
        ins = tok(texts[i:i+bs], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            o = m(**ins)
        v = torch.nn.functional.normalize(o.last_hidden_state[:, 0], dim=-1)
        out.append(v.cpu().numpy().astype("float32"))
    return np.vstack(out)

# ---------- BM25 ----------
def tokenize(s):
    try:
        import jieba
        return [w for w in jieba.cut(s) if w.strip()]
    except Exception:
        return [s[i:i+2] for i in range(len(s)-1)]

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus=corpus; self.k1=k1; self.b=b; self.N=len(corpus)
        self.avgdl=sum(len(d) for d in corpus)/max(1,self.N)
        self.tf=[]; df={}
        for d in corpus:
            f={}
            for w in d: f[w]=f.get(w,0)+1
            self.tf.append(f)
            for w in f: df[w]=df.get(w,0)+1
        self.idf={w: math.log(1+(self.N-c+0.5)/(c+0.5)) for w,c in df.items()}
    def search(self,q,k=20):
        res=[]
        for i,f in enumerate(self.tf):
            s=0.0; dl=sum(f.values())
            for w in q:
                if w in f:
                    tf=f[w]; s+=self.idf.get(w,0)*tf*(self.k1+1)/(tf+self.k1*(1-self.b+self.b*dl/self.avgdl))
            if s>0: res.append((s,i))
        res.sort(key=lambda x:-x[0]); return res[:k]

# ---------- 索引 ----------
class Store:
    def __init__(self):
        self.chunks=json.load(open(os.path.join(DB2,"chunks.json"), encoding="utf-8"))
        import faiss
        self.index=faiss.read_index(os.path.join(DB2,"dense.faiss"))
        self.bm25=BM25(json.load(open(os.path.join(DB2,"bm25_tokens.json"),encoding="utf-8"))["tokens"])
    def dense(self,q,k=20):
        import faiss
        qv=embed([q]); faiss.normalize_L2(qv)
        sc,ix=self.index.search(qv,k)
        return [(float(s),int(i)) for s,i in zip(sc[0],ix[0])]
    def sparse(self,q,k=20):
        return self.bm25.search(tokenize(q),k)

def rrf(lists, k=60):
    sc={}
    for lst in lists:
        for rank,(s,i) in enumerate(lst):
            sc[i]=sc.get(i,0.0)+1.0/(k+rank+1)
    return sorted(sc.items(), key=lambda x:-x[1])

def rewrite(q, n=3):
    """查询改写/扩展：生成同义与拆解查询"""
    p=("你是检索查询改写器。把用户问题改写成 3 条更利于检索的中文查询，"
       "要求：保留原意、补充同义医学表述、必要时拆成子问题。每行一条，不要编号，不要解释。\n用户问题："+q)
    try:
        t=chat([{"role":"user","content":p}], model=ENV.get("RAG_FAST_MODEL","qwen-plus"), max_tokens=200)
        qs=[x.strip(" -•\t") for x in t.splitlines() if x.strip()]
        return [q]+qs[:n]
    except Exception:
        return [q]

_rerank=None
def _reranker():
    global _rerank
    if _rerank is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok=AutoTokenizer.from_pretrained(RERANK_DIR)
        m=AutoModelForSequenceClassification.from_pretrained(RERANK_DIR); m.eval()
        _rerank=(tok,m)
    return _rerank

def rerank(q, cands, top=6):
    """用 bge-reranker 对候选重排（cands: [(score, idx)]）"""
    try:
        import torch
        tok,m=_reranker()
        pairs=[[q, store.chunks[i]["text"]] for _,i in cands]
        with torch.no_grad():
            ins=tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
            s=m(**ins).logits.view(-1).float().cpu().numpy()
        out=sorted([(float(s[j]), cands[j][1]) for j in range(len(cands))], key=lambda x:-x[0])
        return out[:top]
    except Exception as e:
        return cands[:top]

def compress(q, idx, max_sent=4):
    """上下文压缩：从块里挑与问题最相关的句子，降低噪音"""
    t=store.chunks[idx]["text"]
    sents=[s.strip() for s in re.split(r"(?<=[。！？；])", t) if len(s.strip())>=8]
    if len(sents)<=max_sent: return t
    try:
        import faiss
        sv=embed(sents); faiss.normalize_L2(sv)
        qv=embed([q]); faiss.normalize_L2(qv)
        sim=(sv@qv.T).reshape(-1)
        order=sorted(range(len(sents)), key=lambda i:-sim[i])[:max_sent]
        order=sorted(order)
        return "".join(sents[i] for i in order)
    except Exception:
        return t

store=None
def init():
    global store
    if store is None: store=Store()
    return store

def retrieve(q, topk=6, use_rewrite=True, use_rerank=True, use_compress=True):
    init()
    qs = rewrite(q) if use_rewrite else [q]
    lists_d=[store.dense(x,20) for x in qs]
    lists_s=[store.sparse(x,20) for x in qs]
    fused=rrf(lists_d+lists_s)[:20]
    cands=[(s,i) for i,s in fused]
    if use_rerank:
        top=rerank(q, cands, topk)
    else:
        top=cands[:topk]
    out=[]
    for s,i in top:
        c=store.chunks[i]
        txt=compress(q,i) if use_compress else c["text"]
        out.append({"idx":i,"score":s,"source":c["source"],"page":c["page"],"heading":c["heading"],"text":txt})
    return out
