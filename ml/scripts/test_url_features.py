from url_features import (
    extract_url_features,
    URL_FEATURES
)


def main():

    urls = [
        "https://www.google.com",
        "http://192.168.1.100/login",
        "https://secure-paypal-login.example.com/verify?user=123&token=abc",
        "https://facebook.com/example",
    ]

    for url in urls:

        print("=" * 70)

        print(
            f"URL: {url}"
        )

        features = extract_url_features(
            url
        )

        for feature in URL_FEATURES:

            print(
                f"{feature:35} "
                f"{features[feature]}"
            )


if __name__ == "__main__":
    main()