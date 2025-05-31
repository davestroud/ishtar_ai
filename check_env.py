import os
import sys
from dotenv import load_dotenv, find_dotenv

# Attempt to load .env file
print("Ishtar AI Environment Check")
print("---------------------------\\n")

print("1. .env File Loading:")
env_file_path = find_dotenv(usecwd=True)
if env_file_path:
    load_dotenv(env_file_path)
    print(f"  [+] .env file found and loaded from: {env_file_path}\\n")
else:
    print(
        "  [!] .env file not found in the current working directory or parent directories.\\n"
    )
    print(
        "      Please ensure a .env file exists in your project root (e.g., by copying env.example).\\n"
    )


# Helper to display variable status
def check_var(var_name, is_optional=False, is_secret=True):
    value = os.getenv(var_name)
    if value:
        display_value = (
            f"{value[:5]}...{value[-4:]}" if is_secret and len(value) > 9 else value
        )
        print(f"  [+] {var_name:<25} Found ({display_value})")
        return value
    else:
        status = "(Optional)" if is_optional else "(REQUIRED)"
        print(f"  [!] {var_name:<25} Not found {status}")
        return None


print("2. Environment Variables:")
LLM_API_KEY = check_var("LLM_API_KEY")
TAVILY_API_KEY = check_var("TAVILY_API_KEY", is_optional=True)
# ACLED credentials (key required, email recommended)
ACLED_API_KEY = check_var("ACLED_API_KEY", is_optional=True)
ACLED_EMAIL = check_var("ACLED_EMAIL", is_optional=True, is_secret=False)
# LangSmith vars are purely optional for tracing, no direct API test here
check_var("LANGSMITH_API_KEY", is_optional=True)
check_var("LANGCHAIN_TRACING", is_optional=True, is_secret=False)
print("\\n")


print("4. Llama API Connectivity:")
if LLM_API_KEY:
    try:
        from llama_api_client import LlamaAPIClient, APIError

        llama_client = LlamaAPIClient(api_key=LLM_API_KEY)
        # A simple call to list models to verify API key
        models_response = llama_client.models.list()
        if (
            models_response
            and hasattr(models_response, "data")
            and models_response.data
        ):
            print(
                f"  [+] Successfully connected to Llama API. Found {len(models_response.data)} model(s). Example: {models_response.data[0].id}"
            )
        else:
            print(
                f"  [+] Successfully connected to Llama API, but no models listed or unexpected response format."
            )
    except ImportError:
        print("  [!] llama_api_client not installed. Skipping Llama API check.")
        print(
            "      This is part of the project's dependencies. Ensure 'poetry install' was successful."
        )
    except APIError as e:
        print(
            f"  [!] Llama API authentication/connection error: {e.status_code} - {e.message}"
        )
    except Exception as e:
        print(f"  [!] Llama API error: {e}")
else:
    print("  [-] Skipped: LLAMA_API_KEY not set.")
print("\\n")


print("5. Tavily Web Search (Optional):")
if TAVILY_API_KEY:
    try:
        from tavily import TavilyClient

        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        # Perform a minimal search
        search_results = tavily_client.search(
            query="test", max_results=1, search_depth="basic"
        )
        if (
            "results" in search_results and len(search_results["results"]) >= 0
        ):  # Can be 0 if "test" yields nothing
            print(
                "  [+] Successfully connected to Tavily API and performed a test search."
            )
        else:
            print(
                "  [!] Tavily API connected, but test search returned unexpected structure or no results."
            )
    except ImportError:
        print("  [!] tavily-python not installed. Skipping Tavily check.")
        print("      If you intend to use Tavily, run: poetry add tavily-python")
    except Exception as e:
        print(f"  [!] Tavily API error: {e}")
else:
    print("  [-] Skipped: TAVILY_API_KEY not set.")
print("\\n")

print("6. ACLED Data API (Optional):")
if ACLED_API_KEY and ACLED_EMAIL:
    try:
        import httpx

        # Connectivity ping (no credentials required)
        with httpx.Client(timeout=15) as client:
            ping = client.get("https://api.acleddata.com/acled/readme")
            ping.raise_for_status()
            print("  [+] ACLED reachable ✓")

            # Creds present → do a real tiny query
            params = {
                "key": ACLED_API_KEY,
                "email": ACLED_EMAIL,
                "limit": 1,
                "format": "json",
            }
            resp = client.get("https://api.acleddata.com/acled", params=params)
            resp.raise_for_status()
            count = len(resp.json().get("data", []))
            print(f"    [+] Auth OK, {count} test record(s) returned ✓\n")

    except httpx.HTTPStatusError as exc:
        print(
            f"  [!] ACLED request failed: HTTP {exc.response.status_code}: {exc.response.text[:200]}\n"
        )
    except Exception as exc:
        print(f"  [!] ACLED connectivity check error: {exc}\n")
else:
    print("  [-] Skipped: ACLED_API_KEY or ACLED_EMAIL not set.")
print("\\n")

print("Environment check complete.")
print(
    "If any REQUIRED variables are missing or tests failed, please check your .env file and API keys."
)

from ishtar_ai.rag.pipeline import get_vectorstore

vs = get_vectorstore()  # returns a langchain-Pinecone wrapper
docs = vs.similarity_search("Where are Sudanese refugees moving?", k=5)
for d in docs:
    print(d.metadata["source"], d.metadata.get("url"), "⇒", d.page_content[:120])
