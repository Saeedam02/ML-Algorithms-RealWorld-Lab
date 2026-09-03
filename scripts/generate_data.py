"""Generate both reproducible benchmark datasets."""

from ml_realworld.data import generate_and_save

if __name__ == "__main__":
    housing, phishing = generate_and_save()
    print(f"Generated: {housing}")
    print(f"Generated: {phishing}")
