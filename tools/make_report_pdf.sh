#!/usr/bin/env bash
# Build a single shareable PDF of the docs/failure_analysis/ report.
#
# Route: pandoc (md -> standalone HTML) -> headless Chrome (--print-to-pdf).
# Why HTML+Chrome instead of LaTeX/tectonic: LaTeX verbatim does not wrap or scale, so the
# wide ASCII diagrams and tables overflowed/overlapped. Here a small injected script shrinks
# each <pre> just enough to fit the page width, so nothing overlaps and nothing is clipped.
#
# Requires: pandoc + Google Chrome.app (macOS).
#   bash tools/make_report_pdf.sh   ->   FR5_ACT_failure_analysis.pdf
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT="FR5_ACT_failure_analysis.pdf"
WORK="$(mktemp -d -t fa_report)"
COMB="$WORK/combined.md"
HTML="$WORK/report.html"
HEADER="$WORK/header.html"
AFTER="$WORK/after.html"

order=(README 01_symptoms_and_evidence 02_failure_modes_explained 03_training_log_findings \
       04_root_cause_and_fixes 05_glossary_and_references 06_cvae_kl_deep_dive \
       07_grasp_gate_requirements 08_the_fix_implementation 09_model_capacity_vs_overfitting \
       10_free_bits_explained)

: > "$COMB"
for f in "${order[@]}"; do
  cat "docs/failure_analysis/$f.md" >> "$COMB"
  printf '\n\n' >> "$COMB"
done

# --- print stylesheet (goes in <head>) -------------------------------------------------
cat > "$HEADER" <<'CSS'
<style>
  @page { size: A4; margin: 16mm 14mm; }
  html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body {
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
    font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; max-width: none;
  }
  h1 { font-size: 19pt; border-bottom: 2px solid #444; padding-bottom: 4px;
       page-break-before: always; page-break-after: avoid; margin-top: 0; }
  h1:first-of-type { page-break-before: avoid; }
  h2 { font-size: 14pt; margin-top: 1.4em; page-break-after: avoid;
       border-bottom: 1px solid #ddd; padding-bottom: 3px; }
  h3 { font-size: 12pt; page-break-after: avoid; }
  h2, h3, h4, table, pre, blockquote { page-break-inside: avoid; }
  p, li { orphans: 3; widows: 3; }
  a { color: #0b5cad; text-decoration: none; }
  code { font-family: Menlo, Consolas, monospace; font-size: 0.88em;
         background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }
  pre { font-family: Menlo, Consolas, monospace; background: #f6f8fa;
        border: 1px solid #e1e4e8; border-radius: 6px; padding: 10px 12px;
        font-size: 8.6pt; line-height: 1.32; white-space: pre; overflow: hidden; }
  pre code { background: none; padding: 0; font-size: inherit; }
  table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 1em 0; }
  th, td { border: 1px solid #c8ccd0; padding: 5px 8px; text-align: left;
           vertical-align: top; word-break: normal; overflow-wrap: anywhere; }
  th { background: #eef1f4; }
  blockquote { border-left: 4px solid #c7d2dd; margin: 1em 0; padding: 2px 14px;
               color: #333; background: #f8fafc; }
  hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
  /* TOC produced by pandoc --toc */
  #TOC { page-break-after: always; }
  #TOC ul { list-style: none; padding-left: 1em; }
</style>
CSS

# --- fit script (runs in Chrome before print): shrink any <pre> wider than the page ----
cat > "$AFTER" <<'JS'
<script>
  // Scale each code block down just enough that its widest line fits the printable width.
  // Normal-width blocks are untouched; only the wide ASCII diagrams/rules shrink.
  (function () {
    var pres = document.querySelectorAll('pre');
    for (var i = 0; i < pres.length; i++) {
      var pre = pres[i];
      var base = 8.6;                       // matches the CSS font-size (pt)
      var size = base;
      // shrink until the content fits horizontally (down to a 4.2pt floor)
      var guard = 0;
      while (pre.scrollWidth > pre.clientWidth + 1 && size > 4.2 && guard < 40) {
        size -= 0.3; guard++;
        pre.style.fontSize = size.toFixed(2) + 'pt';
      }
    }
    window.__fitDone = true;
  })();
</script>
JS

pandoc "$COMB" -o "$HTML" --standalone --toc --toc-depth=1 \
  --metadata title="FR5 ACT (DINOv2 / DINOv3) — Failure Analysis & Fixes" \
  --include-in-header "$HEADER" --include-after-body "$AFTER"

"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=10000 \
  --print-to-pdf="$OUT" "$HTML" 2>/dev/null

rm -rf "$WORK"
PAGES="$(pdfinfo "$OUT" 2>/dev/null | awk '/Pages/{print $2}')"
echo "wrote $OUT (${PAGES:-?} pages)"
