"""Build the PDF report from docs/report.md (diagrams included).

    pip install -r requirements-dev.txt
    python scripts/build_report.py            # -> docs/Forge_Report.pdf

Steps
  1. expand `<!-- include: path -->` markers in docs/report.md with the referenced doc
     (its H1 dropped, its headings demoted to sit under the including subsection)
  2. convert the Markdown to HTML with a title page, figure captions and print CSS
  3. let headless Edge/Chrome render the Mermaid diagrams and dump the finished DOM
     (the build fails unless every diagram rendered), then print that static page to PDF

Print-only layout tweaks (the Markdown sources stay as they are for GitHub): architecture
flowcharts are laid out top-to-bottom on landscape pages, the state diagram gets a
full page, and sequence diagrams use compact spacing so their text stays readable.
"""
from __future__ import annotations

import datetime as dt
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SOURCE = DOCS / "report.md"
OUTPUT = DOCS / "Forge_Report.pdf"
BUILD_DIR = ROOT / "build"

TITLE = "Forge"
SUBTITLE = "An autonomous, MCP-powered coding agent for the terminal"
AUTHOR = "Toby Wong"
COURSE = "Generative AI · Independent Study (5010)"
MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs"

BROWSERS = [
    os.environ.get("REPORT_BROWSER", ""),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("msedge") or "",
]

ROTATE_MARKER = "as built"     # captions containing this get a rotated (landscape) page
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")


# ---------------------------------------------------------------------------- Markdown -> HTML
def expand_includes(text: str, base: Path) -> str:
    def include(match: re.Match) -> str:
        body = (base / match.group(1).strip()).read_text(encoding="utf-8")
        body = re.sub(r"\A# .*\n", "", body)                       # drop the included doc's title
        return re.sub(r"^(#{2,4}) ", lambda m: "#" * (len(m.group(1)) + 2) + " ", body, flags=re.MULTILINE)
    return re.sub(r"<!--\s*include:\s*(.+?)\s*-->", include, text)


def separate_lists(text: str) -> str:
    """Python-Markdown needs a blank line before a list that follows a paragraph (GitHub doesn't)."""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        prev = out[-1] if out else ""
        if (not in_fence and LIST_ITEM.match(line) and prev.strip() and not LIST_ITEM.match(prev)
                and not prev.startswith((" ", "\t", "|"))):
            out.append("")
        out.append(line)
    return "\n".join(out)


def prepare_markdown(text: str) -> str:
    text = re.sub(r"\A# .*\n", "", text)                            # the title page replaces the H1
    text = re.sub(r"\A\s*(>.*\n)+", "", text)                       # opening note -> dropped from the PDF
    text = re.sub(r"\[([^\]]+)\]\(([^)]+\.md)\)", r"\1", text)      # links to repo docs mean nothing in a PDF
    return separate_lists(text)


def diagram_html(code: str, caption: str, number: int) -> str:
    source = html.unescape(code)
    figure = (f'<pre class="mermaid">{html.escape(source)}</pre>'
              f'<figcaption>Figure {number}. {html.escape(caption)}</figcaption>')
    if source.lstrip().startswith("flowchart") and ROTATE_MARKER in caption.lower():
        # Wide left-to-right diagram: keep its layout and turn it sideways on its own page.
        return f'<figure class="diagram rotated"><div class="rot">{figure}</div></figure>'
    if source.lstrip().startswith("flowchart"):
        source = source.replace("flowchart LR", "flowchart TB", 1)   # taller layout fits a portrait page
        return (f'<figure class="diagram flow"><pre class="mermaid">{html.escape(source)}</pre>'
                f'<figcaption>Figure {number}. {html.escape(caption)}</figcaption></figure>')
    kind = "fullpage" if source.lstrip().startswith("stateDiagram") else "sequence"
    return f'<figure class="diagram {kind}">{figure}</figure>'


def to_html(md_text: str) -> str:
    body = markdown.markdown(prepare_markdown(md_text), extensions=["tables", "fenced_code", "sane_lists"])
    # Number every Mermaid diagram and caption it with the nearest preceding heading.
    parts, last_heading, number, pos = [], "", 0, 0
    pattern = re.compile(r'<h[2-6][^>]*>(.*?)</h[2-6]>|<pre><code class="language-mermaid">(.*?)</code></pre>', re.S)
    for match in pattern.finditer(body):
        if match.group(1) is not None:
            last_heading = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            continue
        number += 1
        parts.append(body[pos:match.start()])
        parts.append(diagram_html(match.group(2), last_heading, number))
        pos = match.end()
    parts.append(body[pos:])
    # Start a new page at the heading that introduces each tall diagram, so the heading,
    # its intro text and the diagram stay together.
    return re.sub(r'<h([2-6])>(?=(?:(?!<h[2-6]).)*?<figure class="diagram (?:sequence|fullpage)")',
                  r'<h\1 class="newpage">', "".join(parts), flags=re.S)


CSS = """
@page { size: Letter; margin: 0.75in 0.75in 0.8in;
        @bottom-right { content: counter(page); font: 9pt 'Segoe UI', sans-serif; color: #888; }
        @bottom-left { content: "Forge · project report"; font: 9pt 'Segoe UI', sans-serif; color: #888; } }
@page :first { @bottom-right { content: none; } @bottom-left { content: none; } }
:root { --brand: #e8742a; --ink: #1f2328; --muted: #5b6470; --rule: #e3e6ea; --code: #f5f6f8; }
* { box-sizing: border-box; }
body { font: 10.5pt/1.5 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; color: var(--ink); margin: 0; background: #fff; }
.title-page { height: 9.3in; display: flex; flex-direction: column; justify-content: center; break-after: page; }
.title-page .bar { width: 64px; height: 6px; background: linear-gradient(90deg, #ffd166, #ff8c42, #d63a1f); margin-bottom: 22px; }
.title-page h1 { font-size: 44pt; margin: 0; letter-spacing: 1px; }
.title-page .subtitle { font-size: 16pt; color: var(--muted); margin: 6px 0 40px; }
.title-page .meta { font-size: 11pt; line-height: 1.8; }
h2 { font-size: 17pt; margin: 0 0 10px; padding-bottom: 6px; border-bottom: 2px solid var(--brand); break-before: page; break-after: avoid; }
h3 { font-size: 13pt; margin: 20px 0 6px; break-after: avoid; }
h4, h5, h6 { font-size: 11pt; margin: 16px 0 4px; color: var(--muted); break-after: avoid; }
p, li { orphans: 3; widows: 3; }
ul, ol { margin: 4px 0 10px; padding-left: 22px; }
li { margin: 2px 0; }
a { color: inherit; text-decoration: none; }
hr { border: none; border-top: 1px solid var(--rule); margin: 14px 0; }
code { font: 9pt Consolas, 'Cascadia Mono', monospace; background: var(--code); padding: 1px 4px; border-radius: 3px; }
pre { background: var(--code); padding: 8px 10px; border-radius: 4px; overflow: hidden; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 14px; font-size: 9pt; }
th, td { border: 1px solid var(--rule); padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #fbf3ec; }
tr { break-inside: avoid; }
figure.diagram { margin: 8px 0 14px; text-align: center; break-inside: avoid; }
figure.diagram pre.mermaid { background: none; padding: 0; margin: 0; white-space: normal; }
figure.diagram svg { max-width: 100% !important; height: auto; }
figure.sequence svg { max-height: 7.9in; }
.newpage { break-before: page; }
figure.flow svg { max-height: 8.6in; }
figure.fullpage svg { max-height: 7.5in; }
figure.rotated { position: relative; height: 9.1in; break-before: page; break-after: page; }
figure.rotated .rot { position: absolute; top: 0; left: 0; width: 9.1in; height: 7in; display: flex;
                      flex-direction: column; align-items: center; justify-content: center;
                      transform-origin: top left; transform: translate(7in, 0) rotate(90deg); }
figure.rotated pre.mermaid { width: 100%; }
figure.rotated svg { width: 100%; max-height: 6.5in; }
figcaption { font-size: 9pt; color: var(--muted); margin-top: 4px; }
"""

MERMAID_CONFIG = """{ startOnLoad: false, theme: 'neutral', securityLevel: 'loose',
  flowchart: { nodeSpacing: 25, rankSpacing: 30, useMaxWidth: true },
  sequence: { actorMargin: 20, messageMargin: 22, boxMargin: 6, width: 110, noteMargin: 6, wrap: true,
              messageFontSize: 14, useMaxWidth: true },
  state: { useMaxWidth: true } }"""


def build_html(body: str) -> str:
    today = dt.date.today().strftime("%B %Y")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{TITLE}: project report</title><style>{CSS}</style></head>
<body>
<section class="title-page">
  <div class="bar"></div>
  <h1>{html.escape(TITLE)}</h1>
  <div class="subtitle">{html.escape(SUBTITLE)}</div>
  <div class="meta"><b>Project report</b><br>{html.escape(AUTHOR)}<br>{html.escape(COURSE)}<br>{today}</div>
</section>
{body}
<script type="module" id="mermaid-loader">
  import mermaid from '{MERMAID_CDN}';
  mermaid.initialize({MERMAID_CONFIG});
  // Render one by one with explicit ids: under headless virtual time Mermaid's own
  // timestamp-based ids can collide, which breaks the later diagrams.
  const blocks = [...document.querySelectorAll('pre.mermaid')];
  for (const [i, el] of blocks.entries()) {{
    const {{ svg }} = await mermaid.render(`diagram-${{i}}`, el.textContent);
    el.innerHTML = svg;
  }}
  document.body.setAttribute('data-rendered', 'true');
</script>
</body></html>"""


# ---------------------------------------------------------------------------- browser steps
def find_browser() -> str:
    browser = next((b for b in BROWSERS if b and Path(b).exists()), None)
    if browser is None:
        sys.exit("No Edge/Chrome found; set REPORT_BROWSER to a Chromium-based browser executable.")
    return browser


def run_browser(browser: str, *args: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as profile:
        return subprocess.run([browser, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", *args],
                              check=True, capture_output=True, timeout=300)


def render_static(browser: str, html_path: Path, expected: int) -> str:
    """Run the page's JavaScript (Mermaid) and return the finished DOM without the script."""
    dom = run_browser(browser, "--virtual-time-budget=120000", "--dump-dom", html_path.as_uri()).stdout
    dom = dom.decode("utf-8", errors="replace")
    rendered = len(re.findall(r'<pre class="mermaid"[^>]*>\s*<svg', dom))
    if 'data-rendered="true"' not in dom or rendered != expected:
        sys.exit(f"Only {rendered}/{expected} diagrams rendered; check your network (Mermaid loads from a CDN).")
    return re.sub(r'<script type="module" id="mermaid-loader">.*?</script>', "", dom, flags=re.S)


def main() -> None:
    BUILD_DIR.mkdir(exist_ok=True)
    body = to_html(expand_includes(SOURCE.read_text(encoding="utf-8"), DOCS))
    expected = body.count('<pre class="mermaid">')
    source_html = BUILD_DIR / "report.html"
    source_html.write_text(build_html(body), encoding="utf-8")

    browser = find_browser()
    static_html = BUILD_DIR / "report_static.html"
    static_html.write_text(render_static(browser, source_html, expected), encoding="utf-8")
    run_browser(browser, "--no-pdf-header-footer", f"--print-to-pdf={OUTPUT}", static_html.as_uri())
    print(f"{expected} diagrams rendered\nHTML: {static_html}\nPDF:  {OUTPUT}")


if __name__ == "__main__":
    main()
