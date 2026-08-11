"""Every branch of cache_text_embeddings must actually WRITE its embedding.

The reference branch — our fork's addition, upstream has no reference plumbing —
encoded a PromptEmbeds and never called .save(). It then set
`is_text_embedding_cached = True`, so the caching pass reported 4/4, spent 32
seconds doing genuine encoding work, wrote zero files, and training died on the
first batch:

    FileNotFoundError: No such file or directory:
    /workspace/data/targets/_t_e_cache/<pair>_HYtdAfLRYEqPRlfYhScwXQ.safetensors

Cost: one pod run. The failure is invisible in the caching pass's own output,
which is what makes it worth a permanent guard rather than a one-line fix.

Checked structurally (AST) rather than by running the loop, because exercising
it for real needs a model, a VAE and a text encoder. The question here is not
"does encoding work" but "does every path reach a save", which is a property of
the source.

Usage:  python testing/test_te_cache_branches.py
"""

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "toolkit", "dataloader_mixins.py")


def _cache_fn():
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "cache_text_embeddings":
            return node
    raise AssertionError("cache_text_embeddings not found")


def _calls(node, attr):
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr]


def main():
    fn = _cache_fn()

    encodes = _calls(fn, "encode_prompt")
    saves = _calls(fn, "save")
    assert encodes, "no encode_prompt calls - did the function move?"
    print(f"  [ok] found {len(encodes)} encode_prompt call(s), "
          f"{len(saves)} save() call(s)")

    # THE INVARIANT: an encode that is not followed by a save in the same block
    # is an embedding computed and thrown away.
    orphans = []
    for parent in ast.walk(fn):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for stmt in body:
            enc = _calls(stmt, "encode_prompt")
            if not enc:
                continue
            # the save may sit in this statement or a later sibling
            idx = body.index(stmt)
            if not any(_calls(s, "save") for s in body[idx:]):
                orphans.append(getattr(stmt, "lineno", "?"))
    assert not orphans, (
        f"encode_prompt with no save() in the same block, line(s) {orphans} - "
        "this is exactly the reference-branch bug that cost a pod run")
    print("  [ok] every encode_prompt is followed by a save() in its block")

    # every branch must consume encode_targets, not just the first pair: an
    # item can carry a DOP variant, and dropping it silently trains on stale text
    target_loops = [n for n in ast.walk(fn)
                    if isinstance(n, ast.For)
                    and isinstance(n.iter, ast.Name) and n.iter.id == "encode_targets"
                    and _calls(n, "encode_prompt")]
    assert len(target_loops) >= 4, (
        f"expected >=4 encode_targets loops (reference / control / first-frame / "
        f"plain), found {len(target_loops)}")
    covered = {id(c) for loop in target_loops for c in _calls(loop, "encode_prompt")}
    stray = [c.lineno for c in encodes if id(c) not in covered]
    assert not stray, (
        f"encode_prompt outside an encode_targets loop at line(s) {stray} - it "
        "encodes one caption and ignores any DOP variant for the same item")
    print(f"  [ok] all {len(encodes)} encodes sit inside "
          f"{len(target_loops)} encode_targets loops")

    # the reference branch specifically, since it is the one that regressed
    ref_branch = None
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and "has_references" in ast.dump(n.test):
            ref_branch = n
            break
    assert ref_branch is not None, "reference branch not found"
    assert _calls(ref_branch, "save"), "reference branch still does not save"
    assert _calls(ref_branch, "cleanup_references"), (
        "reference branch must release its decoded media")
    print("  [ok] reference branch saves, and still cleans up its references")

    print("TEST PASS - every cache_text_embeddings branch writes its embedding")


if __name__ == "__main__":
    main()
