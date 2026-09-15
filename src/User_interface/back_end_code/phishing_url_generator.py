import random
# This combines popular brand names
brands = [
    "google", "paypal", "amazon", "facebook",
    "apple", "microsoft", "netflix", "linkedin",
      "twitter", "whatsapp", "dropbox", "github",
    "yahoo", "spotify", "tiktok", "snapchat",
    "bankofamerica", "chase", "wellsfargo"
]
# social engineering keywords create urgency and fear which are common tactics in phishing attacks 
keywords = [
    "login", "verify", "account", "security",
    "update", "confirm", "billing", "support",
    "password", "authentication","update", "confirm", 
      "secure","alert", "locked", "access", "reset"
]
# often used in phishing URLs to mimic legitimate sites Attackers may use common TLDs to make their URLs appear more trustworthy.
tlds = [
    ".com", ".net", ".org", ".info", ".co", ".support",
    ".xyz", ".online", ".site", ".tech", ".store",
    ".biz", ".live", ".club", ".top"
]

# This function generates a random phishing URL by combining a brand name, two keywords, and a TLD. The resulting URL is designed to mimic the structure of legitimate URLs while incorporating elements commonly found in phishing attempts.
def generate_phishing_url():

    brand = random.choice(brands)
    keyword1 = random.choice(keywords)
    keyword2 = random.choice(keywords)
    tld = random.choice(tlds)
# strings using hyphens to create domain name btrbothom b baad
    domain = f"{brand}-{keyword1}-{keyword2}"

    return f"http://{domain}{tld}"


def generate_urls(n=20):

    urls = []

    for _ in range(n):
        urls.append(generate_phishing_url())

    return urls


if __name__ == "__main__":

    urls = generate_urls(20)

    for u in urls:
        print(u)