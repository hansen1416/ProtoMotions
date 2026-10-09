# Revision validation

6 October 2026.

- All manuscript inputs resolve. No duplicate labels, undefined cross-references, or missing citation keys.
- LaTeX environments and unescaped braces balance in every source file.
- Unseen-body population summaries and figure coordinates recomputed from the retained joined CSV (32,768 pairs).
- Every file under `paper/` has the same SHA-256 as before this revision; no files were added or removed there.
- Built-in compilation of the flattened review source failed before processing the manuscript: `Unable to find standard directories for platform`. No rendered-layout verification or new PDF build is claimed.

`main.tex` and `sections/` are the editable manuscript. `review.tex` is a generated single-file review copy for the built-in editor. Do not edit both copies independently. The existing `main.pdf` is stale and has not been replaced.

Regenerate the review copy by replacing each main-file input with the contents of its referenced `.tex` file. The original multi-file manuscript remains the source of truth.
