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

## The Visual Computer adjustment — 11 October 2026

- review.tex is now the active standalone source, edited in place in the open
  editor. main.tex and sections/ retain the earlier Connection Science pass.
  Do not run build_review.py over the journal-adjusted review.tex.
- The animation framing, RQ1–RQ3, architecture comparison and new unseen-body
  evidence are retained. Reference body-centre floor violations are described
  as such, rather than as measured simulated or mesh penetration.
- Author names/order, affiliation, correspondence, funding, competing interests
  and release access details are explicit placeholders at the user's request.
  All contribution roles are provisionally assigned to Hansen Han for revision.
  A factual AI manuscript-assistance declaration is included.
- With the user's explicit permission, compiled the existing source to a
  temporary layout-check PDF under tmp/pdfs/tvc-layout/. Inspected contact
  sheets for all 34 pages and full-size views of the capability comparison,
  unseen headline, architecture/reference audit and complete-body tables.
  Fixed overflowing lines and wrapped the unseen-table headers. The final
  local compiler log contains no overfull boxes or unresolved references.
- The final source also compiles successfully with the built-in editor compiler.
  The current editor is kept open; no replacement document or tab is created.
- Controlled reference-versus-rollout visuals and simulated sliding/penetration
  measurements remain unavailable. The legacy multibody screenshot lacks the
  checkpoint/clip/body provenance needed to serve as new evaluation evidence.
  No new training, rollout evaluation or synthetic visual evidence was added.

The pre-existing main.pdf remains stale. The temporary PDF is for layout QA;
it is not a submission export or a replacement for the editor preview.

## Screenshot placeholder and project-page synchronization — 11 October 2026

- Added Figure 5, a self-contained screenshot placeholder for one motion
  executed across 128 trained body configurations. Its caption requires the
  final clip, checkpoint and timestamp. No image is presented as new evidence.
- Added the corresponding RQ1 text and updated the discussion and data/code
  availability wording. Used the journal's Statements and Declarations heading.
- Checked the placeholder on page 16 and the affected result pages in the
  temporary PDF. The source remains 34 pages and compiles with the editor.
- Updated C:/Users/hanse/repos/morph_mimic/index.html, style.css and README.md.
  Removed the old 32-body study and added five evidence tables: evaluation
  populations, checkpoint comparison, network comparison, reference audit and
  published capabilities. Retained and labelled the existing trained-body videos.
- Verified the local webpage in the browser, all five tables and internal
  anchor targets; no broken internal anchors were found. Git diff --check passes.
- Published the project-page changes to origin/main in commit d289870.
- Verified the updated tables and removal of the obsolete study on the live
  GitHub Pages URL after deployment; saved a screenshot of the live result table.
- Remaining author-supplied material: final screenshot/provenance, author and
  declaration details, biographies/portraits and release/access identifiers.
  Missing experiments remain limitations, without claims of new results.
