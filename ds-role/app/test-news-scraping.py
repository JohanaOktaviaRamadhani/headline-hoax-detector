import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from scraping.news_scraper import run_scraper


def main():
    print("=" * 50)
    print("NEWS SCRAPER BASED ON HEADLINE")
    print("=" * 50)

    try:
        headline = input("Masukkan headline: ").strip()
    except KeyboardInterrupt:
        print("\n[INFO] Input dibatalkan.")
        return

    if not headline:
        print("Headline tidak boleh kosong.")
        return

    print("\n[INFO] Mencari artikel...\n")

    # 🔥 NO CLEANING
    articles = run_scraper(headline, with_content=True)

    if not articles:
        print("Tidak ditemukan artikel sama sekali.")
        return

    print(f"[INFO] Ditemukan {len(articles)} artikel\n")

    for i, art in enumerate(articles[:10], 1):
        print(f"{i}. {art.get('title', '-')}")
        print(f"   Source   : {art.get('source', '-')}")
        print(f"   URL      : {art.get('url', '-')}")
        print(f"   Published: {art.get('published', '-')}")
        
        content = art.get("content", "")
        snippet = art.get("snippet", "")

        # tampilkan apa adanya
        preview = content if content else snippet

        print(f"   Preview  : {preview[:200]}...")
        print("-" * 50)


if __name__ == "__main__":
    main()