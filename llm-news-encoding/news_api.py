# Purpose: To get the daily news headlines to match the time period of the daily shelter occupancy counts

import requests
import csv

def get_headlines(news_source, offset=0):
    # Get news articles from news API endpoint
    news_api_url = "https://api.worldnewsapi.com/search-news"
    parameters = {
        "api-key": "YOUR_API_KEY",  # change this
        "text": "Toronto",
        "source-countries": "ca",
        "language": "en",
        "news-sources": news_source,
        "earliest-publish-date": "2021-01-01",
        "latest-publish-date": "2023-11-01",
        "sort": "publish-time",
        "sort-direction": "ASC",
        "offset": offset,
        "number": 100  # 100 article count limit per request
    }

    response = requests.get(news_api_url, params=parameters)
    response_obj = response.json()

    while offset < response_obj["available"]:
        if response.status_code == 200:
            if len(response_obj["news"]) > 0:
                # Write news headlines to file
                with open('news_headlines.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    for article in response_obj["news"]:
                        writer.writerow([article["publish_date"], article["title"]])
                    print(f"\r{len(response_obj['news']) + offset} headlines added")
            else:
                print("No news articles found")
        else:
            print(f"Server responded with status code {response.status_code}: {response.reason}. Offset reached {offset}.")

        offset += 100
        parameters["offset"] = offset
        response = requests.get(news_api_url, params=parameters)
        response_obj = response.json()

if __name__ == "__main__":
    get_headlines("https://globalnews.ca")
