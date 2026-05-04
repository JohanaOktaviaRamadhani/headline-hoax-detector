import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://datapers.dewanpers.or.id/site/iframe-verified"

def scrape_page(page):
    params = {
        "page": page,
        "per-page": 10
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(BASE_URL, params=params, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table")
    rows = table.find_all("tr")[1:] 

    data = []

    for row in rows:
        cols = [col.text.strip() for col in row.find_all("td")]

        if len(cols) > 1:
            data.append({
                "nama_media": cols[1],
                "jenis_media": cols[2],
                "penanggung_jawab": cols[3],
                "pemimpin_redaksi": cols[4],
                "badan_hukum": cols[5],
                "provinsi": cols[6],
                "alamat": cols[7],
                "telp": cols[8],
                "email": cols[9],
                "website": cols[10],
                "status": cols[11],
                "tanggal": cols[12],
            })

    return data


# MAIN LOOP
all_data = []
MAX_PAGE = 240
for page in range(1, MAX_PAGE + 1):
    print(f"Scraping page {page}...")

    try:
        data = scrape_page(page)
        if not data:
            break

        all_data.extend(data)
        time.sleep(1) 

    except Exception as e:
        print(f"Error on page {page}: {e}")
        break

# SAVE
df = pd.DataFrame(all_data)
df.to_csv("C:\\Users\\hanao\\Downloads\\hoax-detector\\ds-role\\dataset\\raw\\dewan_pers.csv", index=False)
print("Done. Total data:", len(df))