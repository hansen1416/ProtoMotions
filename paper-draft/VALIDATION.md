# Revision validation — 10 October 2026

- Read the requested planning chat and RQ structure note.
- Downloaded the requested R2 REPORT.md and small evaluation artifacts using
  read-only authenticated requests; no remote objects were changed.
- Inside/outside per-body CSVs each contain 128 unique assets and 233 clips
  per asset. Their rounded means reproduce 97.28 / 96.91% inside and
  97.15 / 97.57% outside for refined / full-data.
- All 256 body rows appear in generated appendix tables. The distribution
  figure includes all bodies, independently ranked per panel.
- Replaced the old unseen-body section, figures, interpretation and provenance.
  Searches of the flattened revised manuscript find no old study numbers or
  severe-case identifiers. The historical paper/ directory is unchanged;
  paper-draft/ is the active revision, as in the previous revision workflow.
- All includes, labels, cross-references and bibliography keys resolve.
- git diff --check passes. The regenerated review.tex compiles successfully
  with the desktop editor compiler. No rendered-layout inspection is claimed.
- No new training or GPU evaluation was performed. This revision integrates
  the recorded results and marks missing evidence explicitly.

main.tex and sections/ are editable sources; build_review.py regenerates the
single-file review.tex and checks references. The desktop editor provides its
PDF preview. The pre-existing main.pdf remains stale; no newly exported PDF
is delivered or claimed.
