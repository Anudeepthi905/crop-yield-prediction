"""Responsive checks in a real browser at phone, tablet, laptop and desktop widths.

Starts the app, then for every page checks: no sideways scrolling, no word broken across lines in
headings, 16 px inputs, and 44 px tap targets on phones. Also checks the Predict flow: on a phone the
result is scrolled into view and focused; from 1024 px inputs and result sit side by side.

Not part of the default run (they need a browser):
    pip install -r requirements-dev.txt && playwright install chromium
    pytest -m e2e
    SAVE_SCREENSHOTS=1 pytest -m e2e      # also refreshes docs/screenshots/
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
SHOTS = ROOT / "docs" / "screenshots"
SAVE = bool(os.environ.get("SAVE_SCREENSHOTS"))
DEVICES = {"phone": (360, 780), "tablet": (768, 1024), "laptop": (1024, 768), "desktop": (1440, 900)}
PAGES = {"home": "/", "predict": "/predict", "about": "/about"}
FIELDS = {"N": "Nitrogen (N), kg/ha", "P": "Phosphorus (P), kg/ha", "K": "Potassium (K), kg/ha"}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = tmp_path_factory.mktemp("streamlit") / "server.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
             "--server.port", str(port), "--browser.gatherUsageStats", "false"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        url = f"http://localhost:{port}"
        for _ in range(90):
            if proc.poll() is not None:
                pytest.fail(f"Streamlit exited early:\n{log_path.read_text()}")
            try:
                urllib.request.urlopen(f"{url}/_stcore/health", timeout=1)
                break
            except OSError:
                time.sleep(1)
        else:
            proc.terminate()
            pytest.fail(f"Streamlit server did not start:\n{log_path.read_text()}")
        yield url
        proc.terminate()
        proc.wait(timeout=20)
    if "Traceback" in log_path.read_text():
        print(log_path.read_text())


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as pw:
        chromium = pw.chromium.launch()
        yield chromium
        chromium.close()


def open_page(browser, server, device, path):
    width, height = DEVICES[device]
    page = browser.new_page(viewport={"width": width, "height": height},
                            is_mobile=device == "phone", has_touch=device in ("phone", "tablet"))
    page.goto(server + path)
    page.wait_for_selector("h1", timeout=60_000)
    if path == "/predict":
        page.wait_for_selector('[data-testid="stNumberInputField"]', timeout=60_000)
    # Wait until Streamlit has finished running the script (its "running" indicator is gone)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=60_000)
    page.wait_for_timeout(800)
    return page


def has_sideways_scroll(page):
    return page.evaluate("""() => {
        const main = document.querySelector('[data-testid="stMain"]');
        return document.documentElement.scrollWidth > innerWidth + 1
            || (main !== null && main.scrollWidth > main.clientWidth + 1);
    }""")


def words_broken_across_lines(page):
    """Words in h1-h3 whose letters are split over two lines (e.g. 'recommendatio / n')."""
    return page.evaluate("""() => {
        const broken = [];
        for (const heading of document.querySelectorAll('h1, h2, h3')) {
            const walker = document.createTreeWalker(heading, NodeFilter.SHOW_TEXT);
            for (let node = walker.nextNode(); node; node = walker.nextNode()) {
                for (const match of node.textContent.matchAll(/\\S+/g)) {
                    const range = document.createRange();
                    range.setStart(node, match.index);
                    range.setEnd(node, match.index + match[0].length);
                    const tops = new Set([...range.getClientRects()].map(r => Math.round(r.top)));
                    if (tops.size > 1) broken.push(match[0]);
                }
            }
        }
        return broken;
    }""")


def small_tap_targets(page):
    return page.evaluate("""() => [...document.querySelectorAll(
            '[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"], '
            + 'button[data-testid^="stBaseButton-primary"], [data-testid="stTooltipHoverTarget"]')]
        .map(el => el.getBoundingClientRect())
        .filter(r => r.width > 0 && (r.width < 43.5 || r.height < 43.5))
        .map(r => `${Math.round(r.width)}x${Math.round(r.height)}`)""")


def predict(page, n=305, p=36, k=69):
    for key, value in zip(FIELDS, (n, p, k)):
        page.locator(f'input[aria-label="{FIELDS[key]}"]').fill(str(value))
    page.get_by_role("button", name="Predict").click()
    page.locator(".st-key-result_card [data-testid=stMetricValue]").first.wait_for(timeout=60_000)
    page.wait_for_timeout(2200)          # smooth scroll, focus and the position re-check finish


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", PAGES)
def test_page_layout(server, browser, name, device):
    page = open_page(browser, server, device, PAGES[name])
    try:
        assert not has_sideways_scroll(page), "page scrolls sideways"
        assert words_broken_across_lines(page) == []
        if name == "predict":
            sizes = page.eval_on_selector_all('[data-testid="stNumberInputField"]',
                                              "els => els.map(e => parseFloat(getComputedStyle(e).fontSize))")
            assert sizes and min(sizes) >= 16, f"input text {sizes} px: iPhone zooms in below 16 px"
            if device == "phone":
                assert small_tap_targets(page) == []
        if name == "home" and device == "phone":
            tops = page.eval_on_selector_all(".st-key-crop_grid [data-testid=stImage]",
                                             "els => els.map(e => Math.round(e.getBoundingClientRect().top))")
            assert len(tops) == 4 and tops[0] == tops[1] and tops[2] == tops[3] and tops[2] > tops[0], \
                f"crop photos should be a 2 x 2 grid on phones, got tops {tops}"
    finally:
        page.close()


def test_phone_result_is_scrolled_into_view_and_focused(server, browser):
    page = open_page(browser, server, "phone", "/predict")
    try:
        predict(page)
        top, bottom = page.evaluate("() => { const r = document.querySelector('.st-key-result_card')"
                                    ".getBoundingClientRect(); return [r.top, r.bottom]; }")
        assert 0 <= top < DEVICES["phone"][1] / 2, f"result card top at {top}px, not brought into view"
        assert page.evaluate("() => document.activeElement.classList.contains('st-key-result_card')")
        assert not has_sideways_scroll(page)
        cut_off = page.eval_on_selector_all(
            ".st-key-result_card [data-testid=stMetricLabel] p, .st-key-result_card [data-testid=stMetricValue]",
            "els => els.filter(e => e.scrollWidth > e.clientWidth + 1).map(e => e.textContent)")
        assert cut_off == [], f"result text cut off on a phone: {cut_off}"
        if SAVE:
            SHOTS.mkdir(parents=True, exist_ok=True)
            page.mouse.move(1, 1)        # no hover toolbar over the chart in the screenshot
            page.wait_for_timeout(500)
            page.screenshot(path=SHOTS / "predict-result-phone.png")
    finally:
        page.close()


@pytest.mark.parametrize("device", ["laptop", "desktop"])
def test_wide_screens_show_inputs_and_result_side_by_side(server, browser, device):
    page = open_page(browser, server, device, "/predict")
    try:
        predict(page)
        form_right = page.evaluate("() => document.querySelector('[data-testid=stForm]').getBoundingClientRect().right")
        card_left = page.evaluate("() => document.querySelector('.st-key-result_card').getBoundingClientRect().left")
        assert card_left > form_right, "result should sit to the right of the inputs"
        if SAVE and device == "desktop":
            SHOTS.mkdir(parents=True, exist_ok=True)
            page.mouse.move(1, 1)
            page.wait_for_timeout(500)
            page.screenshot(path=SHOTS / "predict-result-desktop.png")
    finally:
        page.close()


def test_tablet_is_one_column(server, browser):
    page = open_page(browser, server, "tablet", "/predict")
    try:
        predict(page)
        form_bottom = page.evaluate("() => document.querySelector('[data-testid=stForm]').getBoundingClientRect().bottom")
        card_top = page.evaluate("() => document.querySelector('.st-key-result_card').getBoundingClientRect().top")
        assert card_top > form_bottom, "below 1024 px the result should sit under the inputs"
    finally:
        page.close()


# --- Phone menu ----------------------------------------------------------------

def menu_is_open(page):
    return page.evaluate("""() => { const s = document.querySelector('[data-testid="stSidebar"]');
        if (!s) return false; const r = s.getBoundingClientRect(); return r.width > 0 && r.right > 1; }""")


def open_menu(page):
    page.get_by_test_id("stExpandSidebarButton").click()
    page.wait_for_function("() => { const s = document.querySelector('[data-testid=\"stSidebar\"]');"
                           " return s && s.getBoundingClientRect().right > 1; }", timeout=10_000)


def wait_closed(page):
    page.wait_for_function("() => { const s = document.querySelector('[data-testid=\"stSidebar\"]');"
                           " return !s || s.getBoundingClientRect().right <= 1; }", timeout=10_000)


def test_phone_menu_is_a_hamburger_that_closes_itself(server, browser):
    page = open_page(browser, server, "phone", "/")
    try:
        button = page.get_by_test_id("stExpandSidebarButton")
        # Wait for the menu styles: the hamburger is drawn by ::before (three bars via box-shadow)
        page.wait_for_function("() => { const b = document.querySelector('[data-testid=\"stExpandSidebarButton\"]');"
                               " return b && getComputedStyle(b, '::before').boxShadow !== 'none'; }", timeout=10_000)
        after = button.evaluate("e => getComputedStyle(e, '::after').content")
        size = button.evaluate("e => { const r = e.getBoundingClientRect(); return [r.width, r.height]; }")
        assert after in ("none", "normal"), f"the button should show only the hamburger, not a label ({after})"
        assert size[0] >= 43.5 and size[1] >= 43.5, f"menu button is {size}, under 44 x 44 px"
        assert button.get_attribute("aria-label"), "the button needs an accessible name for screen readers"
        assert page.get_by_test_id("stSidebarLogo").count() == 0
        assert not menu_is_open(page)

        open_menu(page)                                    # choosing a page closes the drawer
        page.get_by_test_id("stSidebarNavLink").filter(has_text="Predict").click()
        page.wait_for_url("**/predict", timeout=30_000)
        wait_closed(page)

        open_menu(page)                                    # Escape closes it
        page.keyboard.press("Escape")
        wait_closed(page)

        open_menu(page)                                    # a tap outside closes it
        page.mouse.click(DEVICES["phone"][0] - 20, DEVICES["phone"][1] // 2)
        wait_closed(page)
    finally:
        page.close()


@pytest.mark.parametrize("device", ["tablet", "desktop"])
def test_wider_screens_show_the_top_navigation(server, browser, device):
    page = open_page(browser, server, device, "/")
    try:
        assert page.get_by_test_id("stExpandSidebarButton").count() == 0
        assert page.get_by_role("link", name="Predict").first.is_visible()
    finally:
        page.close()
