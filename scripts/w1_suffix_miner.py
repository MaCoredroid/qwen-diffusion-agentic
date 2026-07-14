#!/usr/bin/env python3
"""RUNG W-1 drafter — suffix-automaton copy miner (SECTION W, DIRECTIVE-5 + DIRECTIVE-6).

Referenced/adapted from SuffixDecoding (arXiv 2411.04975 / Snowflake) + REST-style retrieval
tries, ported to the FLARE block loop. Pure python, CPU, no torch. Certified by
test_w1_suffix_miner.py (incl. the pointer-slip MINER CERT battery).

DESIGN (DIRECTIVE-6):
  1. Suffix automaton (online Ukkonen SAM) over the LIVE context tokens — longest-suffix-match
     retrieval in O(match) with INCREMENTAL append per turn (agentic contexts grow monotonically;
     append_tokens() never rebuilds).
  2. Longest-match CONTINUATION drafting — draft what historically FOLLOWED the matched suffix
     (extends firing beyond grammar-delimited arg-values into the freetext copy mass).
  3. Recency-first tie-break (DIRECTIVE-5) — among equal-length matches, most-recent source wins.
  4. ANCHORED-MATCH property (the load-bearing safety invariant, DIRECTIVE-6 item 4): every drafted
     continuation begins EXACTLY at the end of an exact suffix match. The off-by-one pointer-slip
     class (continuation sourced from seed_end+1) is therefore STRUCTURALLY UNDRAFTABLE — proven in
     the miner cert. Every candidate is a byte-exact copy of a context slice (copy-assert holds by
     construction).

The miner returns, for a span, the RECENCY-RANKED distinct continuation candidates AND the maximal
common prefix across all longest-match sources. The guard (w1_guard.py) consumes both: the maximal
common prefix is byte-identical regardless of which source is the "right" one, so it is FA-safe to
commit even when the seed is ambiguous (n_cand>=2); divergence beyond it is the ambiguity boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# --------------------------------------------------------------------------------------
# Online suffix automaton (SAM) over the context token stream.
# Standard Ukkonen construction. first_end[state] = an end-position of an occurrence of any
# substring in the state's endpos class; endpos(state) = { first_end of every state in its
# suffix-link subtree }. This is the textbook O(1)-per-state occurrence method.
# --------------------------------------------------------------------------------------
@dataclass
class _State:
    length: int
    link: int
    nxt: dict = field(default_factory=dict)   # token_id -> state_id
    first_end: int = -1                        # end index (inclusive) of the occurrence that created it
    is_clone: bool = False


class SuffixAutomaton:
    """Incremental suffix automaton over a token stream (append-only)."""

    def __init__(self) -> None:
        self.st: list[_State] = [_State(length=0, link=-1)]
        self.last = 0
        self.tokens: list[int] = []              # the live context stream (source of copies)
        # link-subtree children, built lazily for endpos collection
        self._children_dirty = True
        self._children: list[list[int]] = []

    # ---- incremental append (DIRECTIVE-6: no rebuilds mid-episode) ----
    def extend(self, tok: int) -> None:
        pos = len(self.tokens)
        self.tokens.append(tok)
        st = self.st
        cur = len(st)
        st.append(_State(length=st[self.last].length + 1, link=-1, first_end=pos))
        p = self.last
        while p != -1 and tok not in st[p].nxt:
            st[p].nxt[tok] = cur
            p = st[p].link
        if p == -1:
            st[cur].link = 0
        else:
            q = st[p].nxt[tok]
            if st[p].length + 1 == st[q].length:
                st[cur].link = q
            else:
                clone = len(st)
                src = st[q]
                st.append(_State(length=st[p].length + 1, link=src.link,
                                 nxt=dict(src.nxt), first_end=src.first_end, is_clone=True))
                while p != -1 and st[p].nxt.get(tok) == q:
                    st[p].nxt[tok] = clone
                    p = st[p].link
                st[q].link = clone
                st[cur].link = clone
        self.last = cur
        self._children_dirty = True

    def append_tokens(self, toks: Sequence[int]) -> None:
        for t in toks:
            self.extend(t)

    # ---- longest suffix of `query` that is a substring of the context (O(match)) ----
    def longest_suffix_match(self, query: Sequence[int]) -> tuple[int, int]:
        """Return (match_len, state_id) for the longest suffix of `query` occurring in context."""
        st = self.st
        v = 0
        length = 0
        for tok in query:
            if tok in st[v].nxt:
                v = st[v].nxt[tok]
                length += 1
            else:
                while v != -1 and tok not in st[v].nxt:
                    v = st[v].link
                if v == -1:
                    v = 0
                    length = 0
                else:
                    length = st[v].length + 1
                    v = st[v].nxt[tok]
        return length, v

    # ---- endpos (occurrence end indices) of a state, most-recent first, bounded ----
    def _ensure_children(self) -> None:
        if not self._children_dirty:
            return
        n = len(self.st)
        ch: list[list[int]] = [[] for _ in range(n)]
        for i in range(1, n):
            lk = self.st[i].link
            if lk >= 0:
                ch[lk].append(i)
        self._children = ch
        self._children_dirty = False

    def endpos(self, state_id: int, node_budget: int = 20000) -> tuple[list[int], bool]:
        """ALL end indices (inclusive) of occurrences of substrings in this state's class,
        most-recent first. Returns (positions, truncated). node_budget bounds the suffix-link
        subtree walk; if hit, `truncated=True` and the caller MUST treat the span as ambiguous
        (route to K=1) — a partial occurrence list could hide a divergent source, so truncation is
        a hard safety signal, never silently dropped."""
        if state_id <= 0:
            return [], False
        self._ensure_children()
        out: list[int] = []
        stack = [state_id]
        visited = 0
        truncated = False
        while stack:
            s = stack.pop()
            visited += 1
            if visited > node_budget:
                truncated = True
                break
            fe = self.st[s].first_end
            if fe >= 0:
                out.append(fe)
            stack.extend(self._children[s])
        out.sort(reverse=True)          # recency-first: largest end index first
        seen = set()
        uniq = []
        for p in out:                    # de-dup identical positions (clone/original share first_end)
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq, truncated


# --------------------------------------------------------------------------------------
# The drafter: mine recency-ranked continuation candidates + maximal common prefix.
# --------------------------------------------------------------------------------------
@dataclass
class DraftResult:
    match_len: int                       # length of the anchored suffix match (0 => no draft)
    seed: tuple                          # the matched suffix tokens (the anchor)
    candidates: list                     # ALL recency-ranked distinct continuations (tuples, len<=draft_len)
    src_positions: list                  # source START index in context for each candidate (recency-ranked)
    src_dist: list                       # tokens between the draft point and each candidate source, recency order
    common_prefix: tuple                 # maximal prefix shared by ALL longest-match sources (FA-safe to commit)
    n_cand: int                          # number of DISTINCT continuations over ALL sources (true ambiguity)
    truncated: bool = False              # occurrence scan hit the node budget -> treat as ambiguous (K=1)

    @property
    def recency_winner(self) -> tuple:
        return self.candidates[0] if self.candidates else ()

    def verify_candidates(self, cand_cap: int) -> list:
        """The recency-first moving window actually sent to batched verify (DIRECTIVE-5)."""
        return self.candidates[:cand_cap]


class SuffixCopyMiner:
    """Live copy drafter over the growing context. append_context() per turn; draft() per block."""

    def __init__(self, min_match: int = 4, cand_cap: int = 8) -> None:
        self.sam = SuffixAutomaton()
        self.min_match = int(min_match)          # require an anchor >= this many tokens (excludes trivial suffixes)
        self.cand_cap = int(cand_cap)            # moving-window cap on recency-ranked candidates (DIRECTIVE-5)

    # incremental append: prior context + tokens emitted so far this turn (monotonic growth)
    def append_context(self, toks: Sequence[int]) -> None:
        self.sam.append_tokens(toks)

    @property
    def context_len(self) -> int:
        return len(self.sam.tokens)

    def draft(self, emitted_suffix: Sequence[int], draft_len: int) -> DraftResult:
        """Mine a copy draft to fill up to `draft_len` tokens, anchored on `emitted_suffix`
        (the tokens already produced; its longest context-matching suffix is the anchor)."""
        toks = self.sam.tokens
        n = len(toks)
        if draft_len <= 0 or n == 0:
            return DraftResult(0, (), [], [], [], (), 0)
        match_len, state = self.sam.longest_suffix_match(emitted_suffix)
        if match_len < self.min_match:
            return DraftResult(match_len if match_len < self.min_match else 0, (), [], [], [], (), 0)
        seed = tuple(emitted_suffix[-match_len:])
        # ALL occurrence ends of the anchor, recency-first (ambiguity/common-prefix must see every
        # source, not just the recency window — a hidden divergent source is a false-accept risk).
        ends, truncated = self.sam.endpos(state)
        cand_map: dict[tuple, int] = {}      # distinct continuation -> most-recent source START
        for e in ends:
            src_start = e + 1
            cont = tuple(toks[src_start:src_start + draft_len])
            if not cont:
                continue
            if cont not in cand_map:         # ends already recency-sorted => keep the most recent
                cand_map[cont] = src_start
        candidates = list(cand_map.keys())   # ALL distinct continuations, recency-first
        src_positions = [cand_map[c] for c in candidates]
        draft_point = n                       # continuation would be appended at context index n
        src_dist = [draft_point - sp for sp in src_positions]
        common_prefix = self._maximal_common_prefix(candidates)
        return DraftResult(match_len, seed, candidates, src_positions, src_dist,
                           common_prefix, len(candidates), truncated)

    @staticmethod
    def _maximal_common_prefix(cands: list) -> tuple:
        if not cands:
            return ()
        if len(cands) == 1:
            return cands[0]
        out = []
        for i in range(min(len(c) for c in cands)):
            t = cands[0][i]
            if all(c[i] == t for c in cands):
                out.append(t)
            else:
                break
        return tuple(out)


# --------------------------------------------------------------------------------------
# Reference exact-substring miner (the W-0 probe's ad-hoc index) — kept ONLY as a cert
# oracle: the SAM miner must return the same anchored candidate SET.
# --------------------------------------------------------------------------------------
def reference_anchored_candidates(context: Sequence[int], emitted_suffix: Sequence[int],
                                  draft_len: int, min_match: int = 4):
    """Longest exact suffix of emitted_suffix present in context (brute force), recency-first
    ALL distinct continuations (uncapped oracle). Mirrors find_seed_matches() from the W-0 probe."""
    context = list(context)
    n = len(context)
    best_len = 0
    for L in range(min(len(emitted_suffix), n), min_match - 1, -1):
        seed = list(emitted_suffix[-L:])
        if any(context[i:i + L] == seed for i in range(n - L + 1)):
            best_len = L
            break
    if best_len == 0:
        return 0, [], []
    seed = list(emitted_suffix[-best_len:])
    ends = [i + best_len - 1 for i in range(n - best_len + 1) if context[i:i + best_len] == seed]
    ends.sort(reverse=True)
    cand_map = {}
    for e in ends:
        cont = tuple(context[e + 1:e + 1 + draft_len])
        if cont and cont not in cand_map:
            cand_map[cont] = e + 1
    return best_len, list(cand_map.keys()), [cand_map[c] for c in cand_map]


if __name__ == "__main__":
    # tiny smoke
    m = SuffixCopyMiner(min_match=2, cand_cap=8)
    ctx = [10, 11, 12, 13, 14, 99, 10, 11, 12, 20, 21]   # "10 11 12" occurs twice: ->13.. and ->20..
    m.append_context(ctx)
    r = m.draft(emitted_suffix=[50, 10, 11, 12], draft_len=3)
    print("match_len", r.match_len, "n_cand", r.n_cand, "cands", r.candidates,
          "src_dist", r.src_dist, "common_prefix", r.common_prefix, "truncated", r.truncated)
