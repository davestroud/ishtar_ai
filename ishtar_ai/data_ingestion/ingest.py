"""Data ingestion from external sources."""
from typing import List
import httpx

RELIEFWEB_API = "https://api.reliefweb.int/v1/reports"
ACLED_API = "https://api.acleddata.com"
UNHCR_API = "https://api.unhcr.org"

async def fetch_reliefweb(query: str) -> List[dict]:
    params = {"appname": "ishtar-ai", "query[value]": query, "limit": 5}
    async with httpx.AsyncClient() as client:
        resp = await client.get(RELIEFWEB_API, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])

# TODO: implement ACLED and UNHCR fetchers
