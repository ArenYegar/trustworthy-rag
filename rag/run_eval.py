# rag/run_eval.py
import os, sys, json, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import engine, generation
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("init index...", flush=True)
engine.init()
print("index ready", flush=True)
es = json.load(open(os.path.join(base, "data", "eval_set.json"), encoding="utf-8"))
res = {"answerable": [], "unanswerable": []}

def run_one(item, kind):
    q = item["q"]
    t0 = time.time()
    try:
        r = generation.answer(q, topk=6)
    except Exception as e:
        return {"q": q, "error": str(e)[:200]}
    ctx = r.get("contexts", [])
    rec = {"q": q, "status": r.get("status"), "reason": r.get("reason", ""),
           "answer": r.get("answer", "")[:600], "sec": round(time.time()-t0, 1),
           "retrieved": [{"src": c["source"], "pg": c["page"], "score": round(c["score"],3)} for c in ctx],
           "n_ctx": len(ctx)}
    if kind == "answerable":
        kw = item["kw"]
        hit = any(kw in (c["text"] or "") for c in ctx)
        top1 = bool(ctx) and (kw in (ctx[0]["text"] or ""))
        rec["kw"] = kw; rec["kw_in_ctx"] = hit; rec["kw_top1"] = top1
    if r.get("verify"):
        rec["citation_coverage"] = round(r["verify"]["citation_coverage"], 3)
        rec["support_ratio"] = round(r["verify"]["support_ratio"], 3)
    return rec

for it in es["answerable"]:
    rec = run_one(it, "answerable"); res["answerable"].append(rec)
    print("A | %s -> %s | kw_ctx=%s top1=%s" % (it["q"][:24], rec.get("status"), rec.get("kw_in_ctx"), rec.get("kw_top1")), flush=True)
for it in es["unanswerable"]:
    rec = run_one(it, "unanswerable"); res["unanswerable"].append(rec)
    print("U | %s -> %s (%s)" % (it["q"][:24], rec.get("status"), rec.get("reason","")[:40]), flush=True)

out = os.path.join(base, "results_eval.json")
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
# 汇总
A = res["answerable"]; U = res["unanswerable"]
A_ok = sum(1 for r in A if r.get("status")=="answered")
A_hit = sum(1 for r in A if r.get("kw_in_ctx"))
A_top1 = sum(1 for r in A if r.get("kw_top1"))
U_ref = sum(1 for r in U if r.get("status")=="refused")
print("\n==== SUMMARY ====")
print("可答: answered %d/%d | 关键词命中(检索) %d/%d | top1命中 %d/%d" % (A_ok,len(A),A_hit,len(A),A_top1,len(A)))
print("应拒: refused %d/%d" % (U_ref,len(U)))
cc=[r["citation_coverage"] for r in A if "citation_coverage" in r]
sr=[r["support_ratio"] for r in A if "support_ratio" in r]
if cc: print("引用覆盖率 均 %.2f | 支撑率 均 %.2f" % (sum(cc)/len(cc), sum(sr)/len(sr)))
print("DONE", out)
