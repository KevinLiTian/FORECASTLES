# Purpose: To get the daily news headlines to match the time period of the daily shelter occupancy counts

import requests
import csv

if __name__ == "__main__":
    # Get news articles from news API endpoint
    news_api_url = "https://api.worldnewsapi.com/search-news"
    parameters = {
        "api-key": "YOUR_API_KEY",  # change this
        "text": "Toronto prices",
        "source-countries": "ca",
        "language": "en",
        "news-sources": "https://torontosun.com/",
        "earliest-publish-date": "2021-01-01",
        "latest-publish-date": "2023-10-26",
        "sort": "publish-time",
        "sort-direction": "ASC",
        "number": 100  # 100 article count limit per request
    }

    response = requests.get(news_api_url, params=parameters)

    if response.status_code == 200:
        response_obj = response.json()
        if len(response_obj["news"]) > 0:
            # Write news headlines to file
            with open('news_headlines.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                for article in response_obj["news"]:
                    writer.writerow([article["publish_date"], article["title"]])
                print(f"{len(response_obj['news'])} headlines added")
        else:
            print("No news articles found")
    else:
        print(f"Server responded with status code {response.status_code}: {response.reason}")
