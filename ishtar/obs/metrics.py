from prometheus_client import Counter, Histogram

requests_total = Counter("ishtar_requests_total", "Total API requests", ["route"])
retrieval_latency = Histogram("ishtar_retrieval_latency_seconds", "Retriever latency")
