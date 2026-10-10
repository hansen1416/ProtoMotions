"""Flatten manuscript inputs and check references before desktop compilation."""
from pathlib import Path
import re

root = Path(__file__).resolve().parent

def expand(source):
    return re.sub(r"\\input\{([^}]+)\}",
                  lambda m: expand((root / (m[1] + ".tex")).read_text(encoding="utf-8")),
                  source)

source = expand((root / "main.tex").read_text(encoding="utf-8"))
labels = re.findall(r"\\label\{([^}]+)\}", source)
references = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", source)
keys = set(re.findall(r"\\bibitem\{([^}]+)\}", source))
citations = {key for group in re.findall(r"\\cite\{([^}]+)\}", source)
             for key in group.split(",")}
assert len(labels) == len(set(labels)), "Duplicate labels"
assert not set(references) - set(labels), "Undefined cross-references"
assert not citations - keys, "Missing bibliography keys"
(root / "review.tex").write_text(
    "% Generated review copy; edit main.tex and sections.\n" + source, encoding="utf-8")
print("Review regenerated; all inputs, cross-references and citation keys resolve.")
