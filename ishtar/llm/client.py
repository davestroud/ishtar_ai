import httpx
from ishtar.config.settings import settings

def llm_call(prompt: str, stream: bool=False, max_tokens: int = 512, temperature: float = 0.2) -> str:
    # Prefer vLLM/TGI-compatible endpoint if provided
    if settings.vllm_base_url:
        url = f"{settings.vllm_base_url.rstrip('/')}/v1/completions"
        resp = httpx.post(url, json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    # Fallback to OpenAI
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    r = client.completions.create(model="gpt-4o-mini", prompt=prompt, max_tokens=max_tokens, temperature=temperature)
    return r.choices[0].text
