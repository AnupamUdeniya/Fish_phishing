import re
import math
from urllib.parse import urlparse


URL_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "NoOfExternalRef",
]


SOCIAL_DOMAINS = [
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "youtube.com",
    "tiktok.com",
    "reddit.com",
    "pinterest.com",
]


BANK_KEYWORDS = [
    "bank",
    "banking",
    "netbanking",
    "onlinebank",
]


PAY_KEYWORDS = [
    "pay",
    "payment",
    "paypal",
    "checkout",
    "billing",
    "invoice",
]


CRYPTO_KEYWORDS = [
    "crypto",
    "bitcoin",
    "ethereum",
    "wallet",
    "usdt",
    "blockchain",
]


def is_ip_address(domain):

    if not domain:
        return 0

    domain = domain.split(":")[0]

    ipv4_pattern = (
        r"^(25[0-5]|2[0-4][0-9]|"
        r"[01]?[0-9][0-9]?)\."
        r"(25[0-5]|2[0-4][0-9]|"
        r"[01]?[0-9][0-9]?)\."
        r"(25[0-5]|2[0-4][0-9]|"
        r"[01]?[0-9][0-9]?)\."
        r"(25[0-5]|2[0-4][0-9]|"
        r"[01]?[0-9][0-9]?)$"
    )

    return int(
        bool(
            re.fullmatch(
                ipv4_pattern,
                domain
            )
        )
    )


def calculate_entropy(text):

    if not text:
        return 0.0

    frequencies = {}

    for char in text:
        frequencies[char] = (
            frequencies.get(char, 0) + 1
        )

    length = len(text)

    entropy = 0.0

    for count in frequencies.values():

        probability = count / length

        entropy -= (
            probability *
            math.log2(probability)
        )

    return entropy


def extract_url_features(url):

    if not isinstance(url, str):
        url = str(url)

    url = url.strip()

    if not url:
        return {
            feature: 0
            for feature in URL_FEATURES
        }

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        url
    ):
        url = "http://" + url

    parsed = urlparse(url)

    domain = parsed.netloc.lower()

    if "@" in domain:
        domain = domain.split("@")[-1]

    domain = domain.split(":")[0]

    url_lower = url.lower()

    url_length = len(url)

    letters = sum(
        char.isalpha()
        for char in url
    )

    digits = sum(
        char.isdigit()
        for char in url
    )

    special_chars = sum(
        not char.isalnum()
        for char in url
    )

    percent_encoded = len(
        re.findall(
            r"%[0-9a-fA-F]{2}",
            url
        )
    )

    obfuscation = (
        percent_encoded +
        url.count("@") +
        url.count("\\")
    )

    domain_parts = [
        part
        for part in domain.split(".")
        if part
    ]

    subdomain_count = max(
        len(domain_parts) - 2,
        0
    )

    social_net = int(
        any(
            social_domain in domain
            for social_domain
            in SOCIAL_DOMAINS
        )
    )

    bank = int(
        any(
            keyword in url_lower
            for keyword in BANK_KEYWORDS
        )
    )

    pay = int(
        any(
            keyword in url_lower
            for keyword in PAY_KEYWORDS
        )
    )

    crypto = int(
        any(
            keyword in url_lower
            for keyword in CRYPTO_KEYWORDS
        )
    )

    features = {

        "URLLength":
            url_length,

        "DomainLength":
            len(domain),

        "IsDomainIP":
            is_ip_address(domain),

        "NoOfSubDomain":
            subdomain_count,

        "HasObfuscation":
            int(obfuscation > 0),

        "NoOfObfuscatedChar":
            obfuscation,

        "ObfuscationRatio":
            obfuscation /
            max(url_length, 1),

        "NoOfLettersInURL":
            letters,

        "LetterRatioInURL":
            letters /
            max(url_length, 1),

        "NoOfDegitsInURL":
            digits,

        "NoOfEqualsInURL":
            url.count("="),

        "NoOfQMarkInURL":
            url.count("?"),

        "NoOfAmpersandInURL":
            url.count("&"),

        "NoOfOtherSpecialCharsInURL":
            special_chars,

        "SpacialCharRatioInURL":
            special_chars /
            max(url_length, 1),

        "IsHTTPS":
            int(
                parsed.scheme.lower() == "https"
            ),

        "NoOfURLRedirect":
            max(
                url_lower.count("http://") +
                url_lower.count("https://") -
                1,
                0
            ),

        "NoOfSelfRedirect":
            0,

        "HasExternalFormSubmit":
            0,

        "HasSocialNet":
            social_net,

        "HasPasswordField":
            int(
                any(
                    keyword in url_lower
                    for keyword in [
                        "password",
                        "passwd",
                        "pwd"
                    ]
                )
            ),

        "Bank":
            bank,

        "Pay":
            pay,

        "Crypto":
            crypto,

        "NoOfExternalRef":
            0,
    }

    return features


def extract_url_features_dataframe(urls):

    import pandas as pd

    rows = [
        extract_url_features(url)
        for url in urls
    ]

    return pd.DataFrame(
        rows,
        columns=URL_FEATURES
    )