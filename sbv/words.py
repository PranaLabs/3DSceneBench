"""The shared category words only grow. A released room's case and scores depend on two things about
each word it uses: the main word it is scored as and that word's topic (sbv/common/scene.py main_word(), topic()). So a
newer list that keeps both for every released word, and only adds words, moves no released room's scores: a release
that changes only the list needs no room checked again. This compares a released list with a newer one.
Usage: python -m sbv.words [RELEASED] [NEW]
RELEASED defaults to sbv/common/categories_released.txt and NEW to sbv/common/categories.txt. For the list of an older
release by its commit: git show COMMIT:sbv/common/categories.txt > released.txt, then pass released.txt.
Prints each released word that changed and exits 1 if any did."""

import argparse, os
from sbv.common.scene import CATEGORIES, word_groups, word_topics

RELEASED = os.path.join(os.path.dirname(CATEGORIES), "categories_released.txt")


def meanings(path):
    """Each word on a list -> (the main word it is scored as, that main word's topic)."""
    topics = word_topics(path)
    return {w: (m, topics.get(m)) for m, al in word_groups(path).items() for w in [m] + al}


def changes(released=RELEASED, new=CATEGORIES):
    """One line per released word that is gone from the newer list, scored as another word there, or under another
    topic."""
    old, cur = meanings(released), meanings(new)
    out = []
    for w, (m, t) in sorted(old.items()):
        if w not in cur:
            out.append(f"{w}: removed")
            continue
        if cur[w][0] != m:
            out.append(f"{w}: scored as {cur[w][0]}, was {m}")
        if cur[w][1] != t:
            out.append(f"{w}: under topic '{cur[w][1]}', was '{t}'")
    return out


def added(released=RELEASED, new=CATEGORIES):
    """The words on the newer list that the released one does not have."""
    return sorted(set(meanings(new)) - set(meanings(released)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("released", nargs="?", default=RELEASED)
    ap.add_argument("new", nargs="?", default=CATEGORIES)
    a = ap.parse_args(argv)
    bad, new = changes(a.released, a.new), added(a.released, a.new)
    for line in bad:
        print("changed:", line)
    print(
        f"{len(new)} words added, {len(bad)} released words changed: "
        + (
            "the list only grew, so no released room needs checking again"
            if not bad
            else "released rooms must be checked again"
        )
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
