"""
tests/test_core.py  —  QMD core + canonicalize test suite

Run with:  python tests/test_core.py
Or:        python -m pytest tests/ -v  (once pytest is installed)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from itertools import permutations as _perms

from qmd.canonicalize import (
    PermutationCanonicalizer,
    _apply_permutation,
    _lex_key,
    canonical_form,
    are_isomorphic,
    active_backend,
    verify_with_fallback,
)
from qmd.core import (
    to_matrix, to_lists,
    is_skew_symmetric, is_bounded, max_edge,
    mutate,
    quiver_id, mutation_class_id,
    canonical_class_rep,
    explore_mutation_class,
    generate_seed_quivers,
    run_generation,
    MutationClassResult, GenerationResult,
)

# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------
passed = failed = 0
_failures = []

def check(name, condition, msg=''):
    global passed, failed
    if condition:
        passed += 1
        print(f'  PASS  {name}')
    else:
        failed += 1
        detail = f': {msg}' if msg else ''
        print(f'  FAIL  {name}{detail}')
        _failures.append(name)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
A2        = to_matrix([[0, 1],[-1, 0]])
A2_flip   = to_matrix([[0,-1],[ 1, 0]])          # isomorphic to A2
A3        = to_matrix([[0,1,0],[-1,0,1],[0,-1,0]])
A3_rev    = to_matrix([[0,-1,0],[1,0,-1],[0,1,0]])  # A3 reversed — isomorphic
D4        = to_matrix([[0,1,1,1],[-1,0,0,0],[-1,0,0,0],[-1,0,0,0]])
D4_perm   = to_matrix([[0,-1,0,0],[1,0,1,1],[0,-1,0,0],[0,-1,0,0]])  # iso to D4
zero2     = to_matrix([[0,0],[0,0]])
kronecker = to_matrix([[0,2],[-2,0]])

perm_canon = PermutationCanonicalizer()

# ===========================================================================
print('\n=== 1. canonicalize.py — PermutationCanonicalizer ===')
# ===========================================================================

# apply_permutation: identity
check('apply_perm identity', _apply_permutation(A3, (0,1,2)) == A3)

# apply_permutation: reversal
A3_via_perm = _apply_permutation(A3, (2,1,0))
check('apply_perm reversal skew-sym', is_skew_symmetric(A3_via_perm))
check('apply_perm reversal == A3_rev', A3_via_perm == A3_rev)

# canonical_form collapses isomorphic pairs
check('cf: A2 == A2_flip',    perm_canon.canonical_form(A2) == perm_canon.canonical_form(A2_flip))
check('cf: A3 == A3_rev',     perm_canon.canonical_form(A3) == perm_canon.canonical_form(A3_rev))
check('cf: D4 == D4_perm',    perm_canon.canonical_form(D4) == perm_canon.canonical_form(D4_perm))

# canonical_form separates non-isomorphic matrices
check('cf: A2 != A3',         perm_canon.canonical_form(A2) != perm_canon.canonical_form(A3))
check('cf: A3 != D4',         perm_canon.canonical_form(A3) != perm_canon.canonical_form(D4))
check('cf: A2 != zero2',      perm_canon.canonical_form(A2) != perm_canon.canonical_form(zero2))

# canonical_form is idempotent
cf_A3 = perm_canon.canonical_form(A3)
check('cf idempotent A3',     perm_canon.canonical_form(cf_A3) == cf_A3)

# canonical_form output is skew-symmetric
check('cf output skew-sym A3', is_skew_symmetric(perm_canon.canonical_form(A3)))
check('cf output skew-sym D4', is_skew_symmetric(perm_canon.canonical_form(D4)))


# ===========================================================================
print('\n=== 2. canonicalize.py — module-level dispatch ===')
# ===========================================================================

print(f'  INFO  active backend: {active_backend()}')

# Module-level canonical_form must agree with PermutationCanonicalizer
for label, m in [('A2',A2),('A2_flip',A2_flip),('A3',A3),('A3_rev',A3_rev),
                 ('D4',D4),('D4_perm',D4_perm),('zero2',zero2),('kronecker',kronecker)]:
    check(f'dispatch agrees with perm: {label}',
          canonical_form(m) == perm_canon.canonical_form(m))

# are_isomorphic
check('are_isomorphic A2/A2_flip',  are_isomorphic(A2, A2_flip))
check('are_isomorphic A3/A3_rev',   are_isomorphic(A3, A3_rev))
check('are_isomorphic D4/D4_perm',  are_isomorphic(D4, D4_perm))
check('not isomorphic A2/A3',       not are_isomorphic(A2, A3))
check('not isomorphic diff sizes',  not are_isomorphic(A2, A3))

# verify_with_fallback (only meaningful when nauty is active)
if active_backend() == 'nauty':
    for label, m in [('A2',A2),('A3',A3),('D4',D4),('kronecker',kronecker)]:
        check(f'nauty agrees with perm: {label}', verify_with_fallback(m))


# ===========================================================================
print('\n=== 3. Gadget encoding correctness (perm backend validates nauty) ===')
# ===========================================================================
# For the permutation backend (always available), check that the gadget
# encoding round-trips correctly — i.e. two matrices that differ only by
# a permutation produce isomorphic gadget graphs.
# We test this indirectly: if canonical_form gives the same result for
# isomorphic pairs under the perm backend (already tested above), and the
# dispatch backend also agrees, the encoding is correct.

# Additional: weight-2 edge
w2 = to_matrix([[0,2,0],[-2,0,1],[0,-1,0]])
w2_perm = _apply_permutation(w2, (1,0,2))  # swap vertices 0 and 1
check('gadget: weight-2 iso via perm backend',
      perm_canon.canonical_form(w2) == perm_canon.canonical_form(w2_perm))
check('gadget: dispatch agrees on weight-2',
      canonical_form(w2) == perm_canon.canonical_form(w2))

# Non-isomorphic weight-2 matrices differ
w2_diff = to_matrix([[0,2,0],[-2,0,2],[0,-2,0]])
check('gadget: non-iso weight-2 matrices differ',
      canonical_form(w2) != canonical_form(w2_diff))


# ===========================================================================
print('\n=== 4. Matrix helpers ===')
# ===========================================================================
check('to_matrix roundtrip', to_matrix(to_lists(A3)) == A3)
check('is_skew_sym A2',      is_skew_symmetric(A2))
check('is_skew_sym D4',      is_skew_symmetric(D4))
check('not skew_sym',        not is_skew_symmetric(to_matrix([[0,1],[1,0]])))
check('is_bounded A2',       is_bounded(A2, 2))
check('not bounded',         not is_bounded(to_matrix([[0,3],[-3,0]]), 2))
check('max_edge kronecker',  max_edge(kronecker) == 2)


# ===========================================================================
print('\n=== 5. Mutation ===')
# ===========================================================================

# Involution
for label, m in [('A2',A2),('A3',A3),('D4',D4),('kronecker',kronecker)]:
    for k in range(len(m)):
        check(f'involution {label} k={k}', mutate(mutate(m,k),k) == m)

# Skew-symmetry preserved
for label, m in [('A3',A3),('D4',D4)]:
    for k in range(len(m)):
        check(f'mutation skew-sym {label} k={k}', is_skew_symmetric(mutate(m,k)))

check('zero fixed',   all(mutate(zero2,k)==zero2 for k in range(2)))
check('A2 mutate k=0', mutate(A2,0) == A2_flip)
check('A2 mutate k=1', mutate(A2,1) == A2_flip)


# ===========================================================================
print('\n=== 6. ID generation ===')
# ===========================================================================

# quiver_id collapses isomorphic matrices
check('quiver_id A2 == A2_flip',   quiver_id(A2) == quiver_id(A2_flip))
check('quiver_id A3 == A3_rev',    quiver_id(A3) == quiver_id(A3_rev))
check('quiver_id D4 == D4_perm',   quiver_id(D4) == quiver_id(D4_perm))

# quiver_id separates non-isomorphic
check('quiver_id A2 != A3',        quiver_id(A2) != quiver_id(A3))
check('quiver_id A3 != D4',        quiver_id(A3) != quiver_id(D4))

# format checks
qid_a2 = quiver_id(A2)
check('Q.n2 format', qid_a2.startswith('Q.n2.') and len(qid_a2) == len('Q.n2.')+16)
check('Q.n4 format', quiver_id(D4).startswith('Q.n4.'))
check('quiver_id deterministic', quiver_id(A3) == quiver_id(A3))

mcid = mutation_class_id(canonical_form(A2))
check('MC.n2 format', mcid.startswith('MC.n2.') and len(mcid) == len('MC.n2.')+16)
check('mc_id deterministic', mcid == mutation_class_id(canonical_form(A2)))


# ===========================================================================
print('\n=== 7. canonical_class_rep ===')
# ===========================================================================

r_A3 = explore_mutation_class(A3)

# Verify against brute-force computation
manual_min = min(
    (_apply_permutation(m, perm)
     for m in r_A3.labeled_quivers
     for perm in _perms(range(len(m)))),
    key=_lex_key,
)
check('canonical_class_rep == brute force min', r_A3.canonical_rep == manual_min)

# Should be invariant to which isomorphic seed we use
r_A3_rev = explore_mutation_class(A3_rev)
check('mc_id invariant to iso seed (A3/A3_rev)',
      r_A3.mc_id == r_A3_rev.mc_id)

r_D4_perm = explore_mutation_class(D4_perm)
r_D4      = explore_mutation_class(D4)
check('mc_id invariant to iso seed (D4/D4_perm)',
      r_D4.mc_id == r_D4_perm.mc_id)

# Should be invariant to which member of the orbit we start BFS from
mc_id_ref = r_A3.mc_id
for other in r_A3.labeled_quivers:
    if other != A3:
        check('mc_id invariant to orbit member start',
              explore_mutation_class(other).mc_id == mc_id_ref)
        break


# ===========================================================================
print('\n=== 8. BFS exploration — data model ===')
# ===========================================================================

r_A2   = explore_mutation_class(A2)
r_zero = explore_mutation_class(zero2)

# A2: 2 labeled matrices (A2 and its flip are DIFFERENT labeled matrices,
# but the same unlabeled quiver)
check('A2 labeled_size == 2',       r_A2.labeled_size == 2,  f'got {r_A2.labeled_size}')
check('A2 distinct_quiver_count == 1', r_A2.distinct_quiver_count == 1,
      f'got {r_A2.distinct_quiver_count}')
check('A2 not open',                not r_A2.is_open)

# A3: 14 labeled matrices (the full Fomin-Zelevinsky orbit)
check('A3 labeled_size == 14',      r_A3.labeled_size == 14, f'got {r_A3.labeled_size}')
check('A3 distinct_quiver_count >= 1', r_A3.distinct_quiver_count >= 1)
check('A3 not open',                not r_A3.is_open)

# D4: mutation-finite
check('D4 not open',                not r_D4.is_open)
check('D4 labeled_size > 0',        r_D4.labeled_size > 0)

# zero
check('zero labeled_size == 1',     r_zero.labeled_size == 1)
check('zero distinct_quiver_count == 1', r_zero.distinct_quiver_count == 1)
check('zero not open',              not r_zero.is_open)

# quiver_ids list is parallel to labeled_quivers
check('quiver_ids length matches labeled_quivers',
      len(r_A3.quiver_ids) == len(r_A3.labeled_quivers))

# Each quiver_id is the hash of canonical_form of its labeled matrix
check('quiver_ids match canonical_form hash',
      all(quiver_id(m) == qid
          for m, qid in zip(r_A3.labeled_quivers, r_A3.quiver_ids)))

# canonical_rep is skew-symmetric and bounded
check('canonical_rep skew-sym A3',  is_skew_symmetric(r_A3.canonical_rep))
check('canonical_rep bounded A3',   is_bounded(r_A3.canonical_rep))

# All labeled members are bounded
check('all A3 labeled bounded',
      all(is_bounded(m,2) for m in r_A3.labeled_quivers))
check('all D4 labeled bounded',
      all(is_bounded(m,2) for m in r_D4.labeled_quivers))

# Mutation closure: for closed classes, every mutation stays in the orbit
# or exceeds the bound
for label, res in [('A2',r_A2), ('A3',r_A3)]:
    if not res.is_open:
        orbit = set(res.labeled_quivers)
        ok = all(
            mutate(m,k) in orbit or not is_bounded(mutate(m,k),2)
            for m in res.labeled_quivers
            for k in range(len(m))
        )
        check(f'mutation closure {label}', ok)

# boundary_quivers non-empty when is_open (check with a matrix that opens)
m_open = to_matrix([[0,1,1],[-1,0,1],[-1,-1,0]])  # likely opens under bound=1
res_open = explore_mutation_class(m_open, bound=2)
if res_open.is_open:
    check('boundary non-empty when open', len(res_open.boundary_quivers) > 0)


# ===========================================================================
print('\n=== 9. Seed generation ===')
# ===========================================================================

seeds = generate_seed_quivers(max_vertices=4, bound=2)

check('all seeds skew-sym',        all(is_skew_symmetric(s) for s in seeds))
check('all seeds bounded',         all(is_bounded(s,2) for s in seeds))
check('seeds cover n=1..4',        {len(s) for s in seeds} == {1,2,3,4})
check('all seeds in canonical form',
      all(canonical_form(s) == s for s in seeds))

ids = [quiver_id(s) for s in seeds]
check('seed quiver_ids unique', len(ids) == len(set(ids)))

# n=1: only the zero matrix
seeds_n1 = [s for s in seeds if len(s)==1]
check('n=1 count == 1',            len(seeds_n1) == 1)

# n=2: 3 unlabeled quivers  {0}, {±1}, {±2}
seeds_n2 = [s for s in seeds if len(s)==2]
check('n=2 count == 3',            len(seeds_n2) == 3, f'got {len(seeds_n2)}')


# ===========================================================================
print('\n=== 10. Full pipeline ===')
# ===========================================================================

r3 = run_generation(max_vertices=3, bound=2)
r4 = run_generation(max_vertices=4, bound=2)

check('n<=3 quivers non-empty',    len(r3.quivers) > 0)
check('n<=3 classes non-empty',    len(r3.classes) > 0)
check('n<=3 membership non-empty', len(r3.membership) > 0)

# Structural integrity
check('membership keys ⊆ quivers',
      set(r3.membership.keys()) <= set(r3.quivers.keys()))
check('membership values ⊆ classes',
      set(r3.membership.values()) <= set(r3.classes.keys()))

# All stored quivers are in canonical form
check('all quivers canonical n<=4',
      all(canonical_form(m)==m for m in r4.quivers.values()))

# All stored quivers are bounded
check('all quivers bounded n<=4',
      all(is_bounded(m,2) for m in r4.quivers.values()))

# quiver_id keys match hash of stored matrix
check('quiver_id keys consistent n<=3',
      all(quiver_id(m)==qid for qid,m in r3.quivers.items()))
check('quiver_id keys consistent n<=4',
      all(quiver_id(m)==qid for qid,m in r4.quivers.items()))

# mc_id keys match hash of canonical_rep
check('mc_id keys consistent n<=3',
      all(mutation_class_id(mc.canonical_rep)==mcid for mcid,mc in r3.classes.items()))
check('mc_id keys consistent n<=4',
      all(mutation_class_id(mc.canonical_rep)==mcid for mcid,mc in r4.classes.items()))

# All canonical_reps are skew-symmetric and bounded
check('all canon_reps skew-sym',
      all(is_skew_symmetric(mc.canonical_rep) for mc in r4.classes.values()))
check('all canon_reps bounded',
      all(is_bounded(mc.canonical_rep) for mc in r4.classes.values()))

# Full labeled orbits: each class stores its labeled matrices
check('all classes have labeled quivers',
      all(mc.labeled_size > 0 for mc in r4.classes.values()))

# A3 and D4 appear as quivers
check('A3 in n<=4 quivers', quiver_id(A3) in r4.quivers)
check('D4 in n<=4 quivers', quiver_id(D4) in r4.quivers)

# A3 and D4 map to a mutation class
check('A3 has membership',  r4.membership.get(quiver_id(A3)) is not None)
check('D4 has membership',  r4.membership.get(quiver_id(D4)) is not None)

# The A3 mutation class in the pipeline has 14 labeled matrices
a3_mc_id = r4.membership[quiver_id(A3)]
a3_class  = r4.classes[a3_mc_id]
check('A3 class labeled_size == 14',
      a3_class.labeled_size == 14, f'got {a3_class.labeled_size}')

# The D4 mutation class has 50 labeled matrices (known result)
d4_mc_id = r4.membership[quiver_id(D4)]
d4_class  = r4.classes[d4_mc_id]
check('D4 class labeled_size == 50',
      d4_class.labeled_size == 50, f'got {d4_class.labeled_size}')

closed4 = [mc for mc in r4.classes.values() if not mc.is_open]
check('n<=4 has finite-type classes', len(closed4) > 0)
check("total_gluings non-negative", r4.total_gluings >= 0)


# ===========================================================================
print('\n=== 11. Gluing counters and orbit uniqueness theorem ===')
# ===========================================================================

r4 = run_generation(max_vertices=4, bound=2)

# The theorem: closed+closed and closed+open are both bugs — must be 0
check('closed_closed_merges == 0', r4.closed_closed_merges == 0,
      f'got {r4.closed_closed_merges}')
check('closed_open_merges == 0',   r4.closed_open_merges == 0,
      f'got {r4.closed_open_merges}')

# open+open is the only valid case
check('open_open_gluings >= 0',    r4.open_open_gluings >= 0)
check('total_gluings consistent',
      r4.total_gluings == r4.closed_closed_merges
                        + r4.closed_open_merges
                        + r4.open_open_gluings)

# Every merged class (merged_orbit_count > 1) must be open
for mc_id, mc in r4.classes.items():
    if mc.merged_orbit_count > 1:
        check(f'merged class is open ({mc_id[:20]})', mc.is_open)

# Every closed class must have exactly 1 constituent orbit
for mc_id, mc in r4.classes.items():
    if not mc.is_open:
        check(f'closed class has 1 orbit ({mc_id[:20]})',
              mc.merged_orbit_count == 1,
              f'got {mc.merged_orbit_count}')

# --- Synthetic violation tests ---
from qmd.core import _RawOrbit, _UnionFind
from collections import defaultdict

m_shared  = to_matrix([[0, 1], [-1, 0]])
m_other_a = to_matrix([[0, 2], [-2, 0]])
m_other_b = to_matrix([[0,-1], [ 1, 0]])  # different label, same Q.* as m_shared
qid_shared = quiver_id(m_shared)

def _simulate_union(orbits):
    """Run only the union-find step from run_generation and return the error raised, or None."""
    n = len(orbits)
    uf = _UnionFind(n)
    qid_map = defaultdict(list)
    for idx, orb in enumerate(orbits):
        for q in orb.qid_set:
            qid_map[q].append(idx)
    for q, idxs in qid_map.items():
        for i in range(1, len(idxs)):
            a, b = idxs[0], idxs[i]
            if uf.union(a, b):
                a_open = orbits[a].is_open
                b_open = orbits[b].is_open
                if not a_open and not b_open:
                    raise AssertionError(f"closed+closed merge on qid={q!r}")
                elif not a_open or not b_open:
                    raise AssertionError(f"closed+open merge on qid={q!r}")

fake_closed_a = _RawOrbit(
    labeled_quivers=[m_shared, m_other_a],
    quiver_ids=[qid_shared, quiver_id(m_other_a)],
    qid_set={qid_shared, quiver_id(m_other_a)},
    is_open=False, boundary_quivers=[],
)
fake_closed_b = _RawOrbit(
    labeled_quivers=[m_shared, m_other_b],
    quiver_ids=[qid_shared, quiver_id(m_other_b)],
    qid_set={qid_shared, quiver_id(m_other_b)},
    is_open=False, boundary_quivers=[],
)
fake_open = _RawOrbit(
    labeled_quivers=[m_shared], quiver_ids=[qid_shared],
    qid_set={qid_shared}, is_open=True, boundary_quivers=[m_shared],
)

# closed+closed raises
raised_cc = False
try:
    _simulate_union([fake_closed_a, fake_closed_b])
except AssertionError as e:
    raised_cc = "closed+closed" in str(e)
check('closed+closed raises AssertionError', raised_cc)

# closed+open raises
raised_co = False
try:
    _simulate_union([fake_closed_a, fake_open])
except AssertionError as e:
    raised_co = "closed+open" in str(e)
check('closed+open raises AssertionError', raised_co)

# open+open does NOT raise
raised_oo = False
try:
    _simulate_union([fake_open, fake_open])
except AssertionError:
    raised_oo = True
check('open+open does not raise', not raised_oo)

# ---------------------------------------------------------------------------
print(f'\n{"="*62}')
print(f'Results: {passed} passed, {failed} failed')
if _failures:
    print(f'Failed: {", ".join(_failures)}')

print(f'\nPipeline n<=4, bound=2:')
print(f'  Unlabeled quivers:       {len(r4.quivers):>6}')
print(f'  Mutation classes:        {len(r4.classes):>6}')
print(f'    Closed (finite-type):  {len(closed4):>6}')
print(f'    Open:                  {len(r4.classes)-len(closed4):>6}')
print(f'  Inter-class iso merges:  {r4.iso_merges:>6}')
print(f'  Backend: {active_backend()}')
print()
print(f'A3 class: {a3_class.labeled_size} labeled matrices, '
      f'{a3_class.distinct_quiver_count} distinct unlabeled quivers')
print(f'D4 class: {d4_class.labeled_size} labeled matrices, '
      f'{d4_class.distinct_quiver_count} distinct unlabeled quivers')
