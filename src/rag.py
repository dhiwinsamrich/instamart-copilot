"""Part D: chunk the policy documents, TF-IDF retrieval, grounded answers and the retrieval evaluation."""
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mock_llm  # noqa: E402

POLICIES = os.path.join(ROOT, "data", "policies")
SYSTEM = ("You are the Instamart POLICY ASSISTANT. Answer only from the supplied CONTEXT. If the context does not "
          "contain the answer, say: I could not find that information in the policy documents.")


def chunk_documents(size=40, overlap=10):
    chunks = []
    for fn in sorted(os.listdir(POLICIES)):
        words = open(os.path.join(POLICIES, fn), encoding="utf-8").read().split()
        start = 0
        while True:
            chunks.append({"source": fn, "text": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
            start += size - overlap
    return chunks


class Index:
    def __init__(self, size=40, overlap=10):
        self.chunks = chunk_documents(size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(scores[i]), 4),
                 "text": self.chunks[i]["text"]} for i in top]


INDEX = Index()


def retrieve(query, k=3):
    return INDEX.retrieve(query, k)


def answer(question, index=INDEX):
    hits = index.retrieve(question, k=3)
    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    reply = mock_llm.chat(messages).choices[0].message.content
    return reply, [h["source"] for h in hits]


def evaluate(index):
    qa = pd.read_csv(os.path.join(ROOT, "data", "raw", "policy_eval.csv"), keep_default_na=False)
    rows = []
    for q in qa.itertuples(index=False):
        reply, sources = answer(q.question, index)
        if q.expected_doc == "NONE":
            rows.append({"question": q.question, "doc_hit": None, "fact_hit": None,
                         "fallback": reply.startswith("I could not find"), "answer": reply})
        else:
            rows.append({"question": q.question, "doc_hit": q.expected_doc in sources,
                         "fact_hit": q.expected_fact in reply, "fallback": None, "answer": reply})
    return pd.DataFrame(rows)


def report(name, index):
    res = evaluate(index)
    ans, unans = res[res["fallback"].isna()], res[res["fallback"].notna()]
    print(f"\n== evaluation, {name}: {len(index.chunks)} chunks")
    print(f"expected doc in top 3: {int(ans['doc_hit'].sum())}/{len(ans)}; "
          f"answer contains fact: {int(ans['fact_hit'].sum())}/{len(ans)}; "
          f"fallback on unanswerable: {int(unans['fallback'].sum())}/{len(unans)}")
    broken = ans[~(ans["doc_hit"].astype(bool) & ans["fact_hit"].astype(bool))]
    for r in broken.itertuples(index=False):
        print(f"  BROKEN: {r.question!r} doc_hit={r.doc_hit} fact_hit={r.fact_hit} -> {r.answer!r}")
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    counts = pd.Series([c["source"] for c in INDEX.chunks]).value_counts()
    print(f"== chunks (40 words, overlap 10): {len(INDEX.chunks)} total; {counts.to_dict()}")
    print("\n== retrieve('refund policy late delivery')")
    for h in retrieve("refund policy late delivery"):
        print(f"  {h['source']}  {h['score']:.4f}  {h['text'][:70]}...")
    res = report("overlap 10", INDEX)
    print(res[["question", "answer"]].to_string(index=False))
    zero = Index(size=40, overlap=0)
    print(f"\nzero-overlap chunks per doc: {pd.Series([c['source'] for c in zero.chunks]).value_counts().to_dict()}")
    report("overlap 0", zero)
    print("\nzero-overlap chunk boundaries (end of one chunk | start of the next):")
    for a, b in zip(zero.chunks, zero.chunks[1:]):
        if a["source"] == b["source"] in ("refund_policy.txt", "payment_policy.txt"):
            print(f"  [{a['source']}] ...{' '.join(a['text'].split()[-8:])} | {' '.join(b['text'].split()[:8])}...")
