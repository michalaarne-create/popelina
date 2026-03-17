from playwright.sync_api import sync_playwright
import os

# Adres URL strony, którą chcesz otworzyć
TARGET_URL = "https://www.browserscan.net/bot-detection"  # <-- podmień na właściwy adres

# Folder na profil Chrome używany przez Playwrighta
# Możesz zmienić "User" na swoją nazwę użytkownika lub dać inną ścieżkę
USER_DATA_DIR = r"C:\Users\User\playwright-chrome-profile"

# Upewnij się, że katalog istnieje (jeśli nie, zostanie utworzony)
os.makedirs(USER_DATA_DIR, exist_ok=True)

with sync_playwright() as p:
    # Uruchamiamy Chrome w trybie persistent z naszym profilem
    context = p.chromium.launch_persistent_context(
        user_data_dir=USER_DATA_DIR,
        channel="chrome",          # użyj wbudowanego Chrome z Playwrighta
        headless=False,            # żeby widzieć okno
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars"
        ],
        ignore_default_args=["--enable-automation", "--disable-extensions"]
    )

    # Otwieramy nową kartę
    page = context.new_page()

    # Przechodzimy na docelową stronę
    page.goto(TARGET_URL, wait_until="networkidle")

    # Pobieramy HTML i wypisujemy
    html = page.content()
    print(html)

    # Zamykamy przeglądarkę
    context.close()
