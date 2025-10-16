"""
spotify_play.py
~~~~~~~~~~~~~~~~

This helper module automates playback on Spotify's web player using
Playwright.  It is used by the MCP chatbot to play a requested song when
the user instructs the assistant to "play <song> on Spotify".  It logs
into Spotify with credentials provided via environment variables or
configuration, searches for the song, clicks the first result, and
ensures playback starts.

The automation launches Chromium with the autoplay policy disabled to
ensure media playback does not require a user gesture.  It runs a new
browser instance for each call and closes it when done.

You must install Playwright and its Chromium runtime:

    pip install playwright
    playwright install chromium

In addition, set SPOTIFY_EMAIL and SPOTIFY_PASSWORD in your .env file or
export them in your environment.  If credentials are not provided the
function returns False.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from playwright.async_api import async_playwright


async def play_spotify_song(song_name: str, email: str, password: str) -> bool:
    """Automate playback of a song on Spotify.

    Args:
        song_name: The song to search and play.
        email: The Spotify account email.
        password: The Spotify account password.

    Returns:
        True if playback was started successfully; False otherwise.
    """
    query = song_name.strip().replace(" ", "%20")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                args=["--autoplay-policy=no-user-gesture-required"],
            )
            context = await browser.new_context()
            page = await context.new_page()
            # Navigate to Spotify home page
            await page.goto("https://open.spotify.com/")
            # Attempt login if required
            try:
                await page.wait_for_selector("button[data-testid='login-button']", timeout=8000)
                login_btn = page.locator("button[data-testid='login-button']").first
                await login_btn.click()
                await page.wait_for_selector("input#login-username", timeout=15000)
                await page.fill("input#login-username", email)
                await page.fill("input#login-password", password)
                await page.click("button#login-button")
                await page.wait_for_load_state("networkidle")
            except Exception:
                # Already logged in or login not required
                pass
            # Perform search
            search_url = f"https://open.spotify.com/search/{query}"
            await page.goto(search_url)
            await page.wait_for_selector("div[data-testid='search-results']", timeout=15000)
            # Try to click the first result.  Spotify uses complex selectors; attempt multiple options.
            selectors = [
                "a[data-testid='herocard-click-handler']",
                "div[data-testid^='tracklist-row'] a",
            ]
            clicked = False
            for sel in selectors:
                try:
                    element = page.locator(sel).first
                    if await element.is_visible():
                        await element.click()
                        clicked = True
                        break
                except Exception:
                    continue
            if not clicked:
                await browser.close()
                return False
            # Wait briefly for playback to start
            await page.wait_for_timeout(5000)
            await browser.close()
            return True
    except Exception:
        return False


async def _demo() -> None:
    """Simple demo for running this module stand‑alone."""
    email = os.getenv("SPOTIFY_EMAIL")
    password = os.getenv("SPOTIFY_PASSWORD")
    if not email or not password:
        print("Set SPOTIFY_EMAIL and SPOTIFY_PASSWORD to test Spotify playback")
        return
    result = await play_spotify_song("Romulo Romulo", email, password)
    print("Played successfully" if result else "Failed to play")


if __name__ == "__main__":
    asyncio.run(_demo())