from __future__ import annotations

import importlib.util
import subprocess
import time
import unittest
import urllib.request


HAS_PLAYWRIGHT = importlib.util.find_spec("playwright") is not None


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright is not installed")
class StreamlitSmokeTests(unittest.TestCase):
    def test_streamlit_app_renders(self) -> None:
        from playwright.sync_api import sync_playwright

        process = subprocess.Popen(
            [
                "streamlit",
                "run",
                "apps/streamlit/app.py",
                "--server.headless",
                "true",
                "--server.port",
                "8510",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(30):
                try:
                    urllib.request.urlopen("http://localhost:8510", timeout=1)
                    break
                except Exception:
                    time.sleep(1)
            else:
                self.fail("Streamlit app did not start in time.")

            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(viewport={"width": 1400, "height": 1200})
                page.goto("http://localhost:8510", wait_until="domcontentloaded")
                page.wait_for_timeout(4000)
                self.assertGreater(page.locator("text=Traceable LLM Reasoning").count(), 0)
                self.assertGreater(page.locator("text=Run Demo").count(), 0)
                browser.close()
        finally:
            process.terminate()
            process.wait(timeout=10)
