from __future__ import annotations

import json
import sys
from pathlib import Path


def _bar(system_name: str, score: float, index: int) -> str:
    width = max(8, int(score * 520))
    y = 70 + (index * 70)
    label_y = y + 18
    bar_y = y + 26
    percent = f"{score * 100:.1f}%"
    return f"""
  <text x="40" y="{label_y}" font-size="20" fill="#0f172a" font-family="Helvetica, Arial, sans-serif">{system_name}</text>
  <rect x="40" y="{bar_y}" width="520" height="22" rx="11" fill="#e2e8f0" />
  <rect x="40" y="{bar_y}" width="{width}" height="22" rx="11" fill="#2563eb" />
  <text x="580" y="{bar_y + 17}" font-size="18" fill="#0f172a" font-family="Helvetica, Arial, sans-serif">{percent}</text>
"""


def render(summary_path: Path, output_path: Path) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    systems = payload["summary"]
    height = 120 + (len(systems) * 70)
    bars = "\n".join(
        _bar(system_name, systems[system_name]["success_rate"], index)
        for index, system_name in enumerate(systems)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}" viewBox="0 0 760 {height}">
  <rect width="760" height="{height}" fill="#f8fafc" rx="24" />
  <text x="40" y="48" font-size="30" font-weight="700" fill="#020617" font-family="Helvetica, Arial, sans-serif">Benchmark Success Rate</text>
  <text x="40" y="78" font-size="18" fill="#475569" font-family="Helvetica, Arial, sans-serif">{summary_path}</text>
{bars}
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: python scripts/render_summary_svg.py <summary.json> <output.svg>")
        return 1
    render(Path(argv[1]), Path(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
