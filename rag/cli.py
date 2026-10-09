# rag/cli.py  命令行提问：python rag/cli.py "问题"
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import engine, generation

def main():
    q = " ".join(sys.argv[1:]).strip() or input("请输入问题：").strip()
    engine.init()
    r = generation.answer(q)
    print("\n状态：%s%s" % (r["status"], ("（%s）" % r["reason"]) if r.get("reason") else ""))
    print("回答：%s" % r["answer"])
    if r.get("contexts"):
        print("\n引用资料：")
        for i, c in enumerate(r["contexts"], 1):
            pg = ("第%s页" % c["page"]) if c["page"] else ""
            print("  [%d] %s %s" % (i, c["source"], pg))
            print("      %s" % c["text"][:80].replace("\n", " "))
    if r.get("verify"):
        print("\n校验：引用覆盖率 %.2f ｜ 支撑率 %.2f" % (r["verify"]["citation_coverage"], r["verify"]["support_ratio"]))

if __name__ == "__main__":
    main()
