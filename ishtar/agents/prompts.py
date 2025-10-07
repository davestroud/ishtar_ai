SUMMARIZE_TMPL = """You are Ishtar AI. Read the context and answer the user question.
Be concise and cite source titles in brackets where relevant.

Question:
{query}

Context (JSON-like lines of doc snippets & metadata):
{context}
"""

VERIFY_TMPL = """You are a fact checker. Compare the draft with the context.
Return a short verdict: 'OK' if supported by the context, otherwise list specific issues.

Draft:
{draft}

Context:
{context}
"""

REFINE_TMPL = """Refine the draft based on the verifier's verdict.
If 'OK', polish wording. If there are issues, correct them using only the context.

Draft:
{draft}

Verdict:
{verdict}
"""
