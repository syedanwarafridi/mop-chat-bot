import requests
import os
from twitter_apis import get_my_tweets_and_replies
import psycopg2
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
    
# -----------------------> Token API BASE API <----------------------- #
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
            return {
                "coinmarketcap_update": coinmarketcap_update,
                "solana_tracker_update": solana_tracker_update
            }
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

# -----------------------> Tavily API for replies <----------------------- #
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_api_key = os.getenv('TAVILY_API_KEY')

os.environ['TAVILY_API_KEY'] = tavily_api_key
def tavily_data(query: str):
    tool = TavilySearchResults(max_results=5,include_domains=["https://crypto.news/", "https://cointelegraph.com/", "https://dexscreener.com/"], include_images=False, include_videos=False, include_links=True)
    results = tool.invoke(query)
    # filtered_results = [{"title": item["title"], "content": item["content"]} for item in results]
    return results

# -----------------------> Tavily google search <----------------------- #
def google_search(query: str):
    tool = TavilySearchResults(max_results=3, include_images=False, include_videos=False, include_links=True)
    results = tool.invoke(query)
    return results

# -----------------------> Tavily for POSTs <----------------------- #
tavily_api_key = os.getenv('TAVILY_API_KEY')

os.environ['TAVILY_API_KEY'] = tavily_api_key
def tavily_for_post(query: str):
    tool = TavilySearchResults(max_results=1, include_domains=["https://www.reuters.com/markets/cryptocurrency/", "https://www.forbes.com/digital-assets/news/?sh=487b1daf9d5b", "https://finance.yahoo.com/topic/crypto/", "https://crypto.news/", "https://finance.yahoo.com/markets/"], include_images=False, include_videos=False, include_links=True)
    # tools = [tool]
    results = tool.invoke(query)
    # filtered_results = [{"title": item["title"], "content": item["content"]} for item in results]
    return results

# -----------------------> RAG DB STATS <----------------------- #
def get_rag_db_stats():
    db_params = {
        'host': '147.93.40.59',
        'port': 5432,
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'vrHwHn0VNZq5h7z2nbe4bZworOaW5GEr'
    }

    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT SUM(reltuples)::BIGINT AS total_rows
            FROM pg_class
            WHERE relkind = 'r';
        """)
        total_rows = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM crypto_assets_embeddings;")
        total_news_items = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM trending_tokens;")
        total_trending_tokens = cursor.fetchone()[0]

        last_news_update = cursor.fetchone()
        last_news_update = last_news_update[0] if last_news_update else None

        cursor.close()
        conn.close()

        stats = {
            'total_rows_in_rag_db': total_rows,
            'total_news_items_processed': total_news_items,
            'total_trending_tokens_processed': total_trending_tokens,
        }

        return stats

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return None

# -----------------------> Full STATS <----------------------- #
def get_combined_stats_with_api():
    twitter_data = get_my_tweets_and_replies()
    db_data = get_rag_db_stats()

    api_data = last_update_api()

    if twitter_data is None or db_data is None or "error" in api_data:
        print("Error retrieving data from one or more sources.")
        return None

    combined_data = {**twitter_data, **db_data, **api_data}
    return combined_data
