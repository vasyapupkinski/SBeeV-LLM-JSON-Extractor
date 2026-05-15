from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        def handle_response(response):
            if "json" in response.headers.get("content-type", "") and "zighang.com" in response.url:
                print(f"API FOUND: {response.url}")
                
        page.on("response", handle_response)
        print("Navigating to Zighang...")
        page.goto("https://zighang.com/recruitment")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(3000)
        browser.close()

if __name__ == "__main__":
    run()
