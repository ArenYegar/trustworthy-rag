# rag/generation.py  --  句级引用 + 支撑校验 + 拒答闸门
import os, re, json
from . import engine
from .engine import chat, ENV, retrieve

CITE_PAT = re.compile(r"\[(\d+)\]")
REFUSE_THRESHOLD = float(os.environ.get("RAG_REFUSE_THRESHOLD", "0.30"))
SUPPORT_MIN = float(os.environ.get("RAG_SUPPORT_MIN", "0.60"))
COVER_MIN = float(os.environ.get("RAG_COVER_MIN", "0.20"))  # 覆盖率只作提示，主闸门是支撑率
REFUSE_TEXT = "知识库中没有找到足够的依据来回答该问题，为避免误导不作猜测。"

def build_prompt(q, contexts):
    parts = []
    for i, c in enumerate(contexts, 1):
        pg = c.get("page")
        tag = "[%d] 来源：%s%s" % (i, c.get("source", ""), (" 第%s页" % pg) if pg else "")
        parts.append(tag + "\n" + c["text"])
    refs = "\n\n".join(parts)
    return (
        "你是严谨的心理健康知识科普助手。只能依据【参考资料】作答，禁止使用任何外部知识。\n"
        "规则：\n"
        "1) 每一句话（含概括句、过渡句）都必须以引用标记 [n] 结尾，n 是资料编号；不要写不带引用的句子；\n"
        "2) 资料中没有依据的内容一律不写；若资料完全无法回答该问题，只输出两个字：无依据；\n"
        "3) 不做诊断、不给具体用药剂量、不替代就医；若涉及自伤/自杀风险，提示立即联系专业机构或紧急求助。\n"
        "【参考资料】\n" + refs + "\n\n【问题】\n" + q + "\n\n【回答】\n"
    )

def generate(q, contexts, model=None):
    return chat([{"role": "user", "content": build_prompt(q, contexts)}],
                model=model or ENV.get("RAG_MODEL", "qwen-plus"), max_tokens=800)

def split_sentences(t):
    return [s.strip() for s in re.split(r"(?<=[。！？；])", t.strip()) if s.strip()]

def verify(answer, contexts, model=None):
    claims = []
    for s in split_sentences(answer):
        cn = [int(x) for x in CITE_PAT.findall(s)]
        cn = [n for n in cn if 1 <= n <= len(contexts)]
        body = CITE_PAT.sub("", s).strip()
        if len(body) >= 6:
            claims.append({"sent": body, "cites": cn})
    n = len(claims)
    if n == 0:
        return {"sentences": 0, "cited": 0, "supported": 0, "citation_coverage": 0.0,
                "support_ratio": 0.0, "details": []}
    cited = sum(1 for c in claims if c["cites"])
    blocks = []
    for k, c in enumerate(claims):
        if not c["cites"]: continue
        ctx = "\n".join("[%d] %s" % (x, contexts[x-1]["text"]) for x in c["cites"])
        blocks.append("第%d条\n论断：%s\n资料：%s" % (k, c["sent"], ctx))
    supported = set()
    if blocks:
        prompt = ("你是事实核查员。对每组「论断 + 资料」，判断资料是否支撑该论断。"
                  "严格只输出一个 JSON 数组，元素形如 {\"k\":0,\"ok\":true}，不要任何多余文字。\n\n"
                  + "\n\n".join(blocks))
        try:
            t = chat([{"role": "user", "content": prompt}],
                     model=model or ENV.get("RAG_FAST_MODEL", "qwen-plus"), max_tokens=600)
            got = False
            m = re.search(r"\[.*\]", t, re.S)
            if m:
                try:
                    for it in json.loads(m.group(0)):
                        if isinstance(it, dict) and it.get("ok"): supported.add(int(it.get("k")))
                    got = True
                except Exception:
                    got = False
            if not got:  # 宽松回退：按行取 "k ... 是/支持/true"
                for ln in t.splitlines():
                    mm = re.search(r"(\d+)\D{0,6}?(是|支持|true|yes|ok)", ln, re.I)
                    if mm: supported.add(int(mm.group(1)))
        except Exception:
            pass
    details = [{"sent": c["sent"][:90], "cites": c["cites"], "supported": (k in supported)}
               for k, c in enumerate(claims)]
    return {"sentences": n, "cited": cited, "supported": len(supported),
            "citation_coverage": cited / n,
            "support_ratio": (len(supported) / cited) if cited else 0.0,
            "details": details}

def answer(q, topk=6, use_rewrite=True, use_rerank=True):
    ctx = retrieve(q, topk=topk, use_rewrite=use_rewrite, use_rerank=use_rerank)
    if not ctx:
        return {"query": q, "status": "refused", "reason": "无检索结果", "answer": REFUSE_TEXT, "contexts": []}
    if ctx[0]["score"] < REFUSE_THRESHOLD:
        return {"query": q, "status": "refused", "reason": "检索相关度过低 %.3f" % ctx[0]["score"],
                "answer": REFUSE_TEXT, "contexts": ctx}
    raw = generate(q, ctx)
    if raw[:8].replace(" ", "").startswith("无依据"):
        return {"query": q, "status": "refused", "reason": "模型判定无依据", "answer": REFUSE_TEXT, "contexts": ctx}
    ver = verify(raw, ctx)
    if ver["support_ratio"] < SUPPORT_MIN:
        return {"query": q, "status": "refused",
                "reason": "支撑不足(支持率%.2f)" % ver["support_ratio"],
                "answer": REFUSE_TEXT, "contexts": ctx, "verify": ver}
    return {"query": q, "status": "answered", "answer": raw, "contexts": ctx, "verify": ver}
