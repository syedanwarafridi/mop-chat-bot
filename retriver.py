import requests
import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------> Similarity/Distance BASE API <----------------------- #
def distance_api(query: str):
    BASE_URL = "https://mop.rekt.life/v1/query"
    PARAMS = {"query": query}
    
    try:
        response = requests.get(BASE_URL, params=PARAMS)

        if response.status_code == 200:
            response = response.json()
            top_items = sorted(response["data"], key=lambda x: x['distance'])[:3] 
            return [item for item in top_items]
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    
# -----------------------> Similarity/Distance BASE API <----------------------- #
def token_api(query: str):
    url = "http://mop.rekt.life/v1/search"
    try:
        response = requests.get(url, params={"query": query})
        
        if response.status_code == 200:
            return response.json()
        else:
            # return {"error": f"Request failed with status code {response.status_code}", "details": response.text}
            return {"data": f" "}
    
    except requests.exceptions.RequestException as e:
        return {"error": "Request exception occurred", "details": str(e)}
    
# -----------------------> Last Update API <----------------------- #
def last_update_api():
    url = "https://mop.rekt.life/v1/update/crypto_assets"
    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        if isinstance(data, dict) and data.get("success") and isinstance(data.get("data"), list) and len(data["data"]) == 2:
            coinmarketcap_update = data["data"][0].get("last_update", "N/A")
            solana_tracker_update = data["data"][1].get("last_update", "N/A")
            return f"Last Update for Coinmarketcap: {coinmarketcap_update}, \nLast Update for SolanaTracker: {solana_tracker_update}"
        else:
            return {"error": "Unexpected JSON structure or missing data."}
    except requests.exceptions.RequestException as e:
        return {"error": "Request exception occurred", "details": str(e)}
    except ValueError:
        return {"error": "Failed to parse JSON response."}
    
# -------------------------> Update Data API <----------------------- #
def update_api():
    url = "https://mop.rekt.life/v1/update/crypto_assets"
    try:
        response = requests.post(url)
        
        if response.status_code == 200:
            
            return response.json()
        else:
            return {"error": f"Request failed with status code {response.status_code}", "details": response.text}
    
    except requests.exceptions.RequestException as e:
        return {"error": "Request exception occurred", "details": str(e)}

# -----------------------> Tavily API <----------------------- #
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_api_key = os.getenv('TAVILY_API_KEY')

os.environ['TAVILY_API_KEY'] = tavily_api_key
def tavily_data(query: str):
    tool = TavilySearchResults(max_results=5,include_domains=["https://crypto.news/", "https://cointelegraph.com/", "https://dexscreener.com/"], include_images=False, include_videos=False, include_links=True)
    # tools = [tool]
    results = tool.invoke(query)
    # filtered_results = [{"title": item["title"], "content": item["content"]} for item in results]
    return results

# -----------------------> Tavily for POSTs <----------------------- #
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_api_key = os.getenv('TAVILY_API_KEY')

os.environ['TAVILY_API_KEY'] = tavily_api_key
def tavily_for_post(query: str):
    tool = TavilySearchResults(max_results=10, include_domains=["https://www.reuters.com/markets/cryptocurrency/", "https://www.forbes.com/digital-assets/news/?sh=487b1daf9d5b", "https://finance.yahoo.com/topic/crypto/", "https://crypto.news/", "https://finance.yahoo.com/markets/"], include_images=False, include_videos=False, include_links=True)
    # tools = [tool]
    results = tool.invoke(query)
    # filtered_results = [{"title": item["title"], "content": item["content"]} for item in results]
    return results