# interest-memory

Learns what a reader likes from weighted feedback and scores new text by it. Uses local embeddings (fastembed) and SQLite. Makes no LLM calls.

```python
mem = InterestMemory("memory.db")
mem.add_seeds(["agent tool use", "RAG evaluation"])
mem.record(item_id, text, weight, at)   # e.g. +0.15 kept, -1.0 rejected
mem.forget(item_id)                     # undo
mem.score(texts, now)                   # weighted kNN with decay
mem.profile() / mem.set_profile(text)
```

Test: `uv run pytest`
