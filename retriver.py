import requests
import os

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
            return {"data": f"Data is not available for the token."}
    
    except requests.exceptions.RequestException as e:
        return {"error": "Request exception occurred", "details": str(e)}
    

# -----------------------> Tavily API <----------------------- #
# from langchain_community.tools.tavily_search import TavilySearchResults

# os.environ['TAVILY_API_KEY'] = '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'
# def tavily_data(query: str):
#     tool = TavilySearchResults(max_results=5)
#     # tools = [tool]
#     results = tool.invoke(query)
#     filtered_results = [{"title": item["title"], "content": item["content"]} for item in results]
#     return filtered_results


# token_address = "0xdc7F7Ad44f3Ed116577417017cf37a19DFf9FFe9"
# result = token_api(token_address)
# print(result)
