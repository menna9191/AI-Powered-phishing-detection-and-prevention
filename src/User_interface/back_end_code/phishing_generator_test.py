import requests
from phishing_url_generator import generate_urls

API = "http://127.0.0.1:5000/detect"

urls = generate_urls(20)

for url in urls:

    response = requests.post(
        API,
        json={"url": url}
    )

    result = response.json()

    print("URL:", url)
    print("Prediction:", result["prediction"])
    print("Probability:", result["probability"])
    print("-------------")