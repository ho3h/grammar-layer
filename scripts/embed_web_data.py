"""Inject the latest web/data/summary.json INTO web/index.html as a fallback
data island. Browsers that can't fetch (sandboxed previews, file:// loads) will
fall back to this embedded copy.

This is idempotent — re-running just replaces the existing data island.

Usage:
    uv run python scripts/embed_web_data.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from neograph.config import PATHS


HTML_PATH = PATHS.root / "web" / "index.html"
DATA_PATH = PATHS.root / "web" / "data" / "summary.json"

START = "<!-- BEGIN summary-island -->"
END = "<!-- END summary-island -->"

# We escape inner script tags safely by closing with </​script> and replacing
# any sequence that could close the embedded block.
SCRIPT_OPEN = '<script id="summary-island" type="application/json">'
SCRIPT_CLOSE = "</script>"


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} does not exist. Run scripts/export_web_data.py first.",
              file=sys.stderr)
        return 1
    if not HTML_PATH.exists():
        print(f"ERROR: {HTML_PATH} does not exist.", file=sys.stderr)
        return 1

    summary = DATA_PATH.read_text()
    # Defang any literal "</script>" inside the JSON (extremely unlikely but cheap).
    summary = summary.replace("</", "<\\/")

    island = f"{START}\n{SCRIPT_OPEN}\n{summary}\n{SCRIPT_CLOSE}\n{END}"

    html = HTML_PATH.read_text()
    if START in html and END in html:
        # Use a lambda replacement so re.sub doesn't interpret backslash
        # sequences in the JSON payload (e.g. A) as regex backreferences.
        new_html = re.sub(
            re.escape(START) + r".*?" + re.escape(END),
            lambda _m: island,
            html,
            count=1,
            flags=re.DOTALL,
        )
    else:
        # Insert just before </body>
        new_html = html.replace("</body>", f"  {island}\n</body>", 1)

    HTML_PATH.write_text(new_html)
    print(f"Injected {len(summary)} bytes of summary into {HTML_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
