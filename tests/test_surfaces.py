"""Marked surfaces, their triangulations, and the quivers they carry."""

import pytest

from qmd import dynkin, surfaces
from qmd.core import _bfs_unlabeled, _lex_key, mutation_class_id


def _mc_id(t, matching):
    q = surfaces.quiver_of(t, matching)
    orbit = _bfs_unlabeled(q, 2, None)
    return mutation_class_id(min(orbit.members, key=_lex_key))


def test_euler_formula_holds_for_every_generated_triangulation():
    """
    The whole module rests on reading the surface off the gluing, so check the
    two descriptions agree: the arc count from the matching must equal the one
    the FST formula predicts from (g, b, p, c).
    """
    for n in range(3, 8):
        for sig, (t, matching) in surfaces.triangulations_for(n, samples=20_000).items():
            g, b, p, m = sig
            c = sum(m)
            assert 6 * g + 3 * b + 3 * p + c - 6 == n, (n, sig)
            assert len(matching) // 2 == n, (n, sig)
            assert 3 * t == 2 * n + c, (n, sig, t)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_every_surface_of_the_rank_is_realised(n):
    """A missed surface would silently drop a class from the reference table."""
    found = set(surfaces.triangulations_for(n, samples=60_000))
    missing = surfaces.enumerate_signatures(n) - found
    assert not missing, sorted(surfaces.name_of(s) for s in missing)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_polygons_are_type_a_and_punctured_polygons_are_type_d(n):
    """
    Independent cross-check against qmd.dynkin: the (n+3)-gon must land on
    A_n and the once-punctured (n)-gon on D_n. Two unrelated constructions
    agreeing on the same mc_id is the strongest evidence the gluing model and
    the arrow convention are right.
    """
    tri = surfaces.triangulations_for(n, samples=60_000)
    ref = dynkin.reference_for(n)
    poly = _mc_id(*tri[(0, 1, 0, (n + 3,))])
    assert ref.get(poly) == f"A{n}", (n, poly, ref.get(poly))
    if n >= 4:
        punctured = _mc_id(*tri[(0, 1, 1, (n,))])
        assert ref.get(punctured) == f"D{n}", (n, punctured, ref.get(punctured))


def test_once_punctured_torus_is_the_markov_class():
    """The rank-3 once-punctured torus is the curated Markov class."""
    tri = surfaces.triangulations_for(3, samples=20_000)
    assert _mc_id(*tri[(1, 0, 1, ())]) == "MC.n3.7405511b230b7552"


def test_rank_6_surface_classes_are_exactly_the_census_ones():
    """
    Rank 6 has thirteen mutation-finite classes. Eleven come from surfaces;
    the other two are the exceptional E6 and X6, and X6 in particular must NOT
    be produced by any surface — Derksen-Owen proved it is not block
    decomposable, so a surface hitting it would mean this module is wrong.
    """
    tri = surfaces.triangulations_for(6, samples=60_000)
    ids = {_mc_id(t, m) for t, m in tri.values()}
    assert len(tri) == 11 and len(ids) == 11
    assert "MC.n6.03e32ee9eeb2b09e" not in ids, "X6 does not come from a surface"
    assert "MC.n6.63a47ebf7e805edc" not in ids, "E6 does not come from a surface"
    expected = {
        "MC.n6.0a46ca485a0497d7",   # 9-gon (A6)
        "MC.n6.88cf05708ebd18ae",   # once-punctured 6-gon (D6)
        "MC.n6.28d1d4164324e327",   # annulus(1,5)
        "MC.n6.114aa9334ddab0b4",   # annulus(2,4)
        "MC.n6.64b3aad6b780ddba",   # annulus(3,3)
        "MC.n6.71357003153649ba",   # twice-punctured 3-gon
        "MC.n6.3652dd6e93335d5a",   # once-punctured annulus(1,2)
        "MC.n6.8f733ca729627016",   # torus, 1 boundary, 3 marked
        "MC.n6.97aa24facb7e6e72",   # pair of pants(1,1,1)
        "MC.n6.299c7f0892f2765a",   # twice-punctured torus
        "MC.n6.45cb86cc44e98cd3",   # 4-punctured sphere
    }
    assert ids == expected


def test_classical_names_agree_with_dynkin():
    for n in range(3, 8):
        for sig in surfaces.enumerate_signatures(n):
            name = surfaces.classical_name(sig)
            if name and name.startswith("A") and "~" not in name:
                assert name == f"A{n}"
            if name and name.startswith("D") and "~" not in name:
                assert name == f"D{n}"


def test_export_labels_a_surface_class_without_any_curation():
    """
    The point of the whole module: a class that is not a Dynkin type still
    comes out named, so a new rank needs no hand-curation. Rank 4's
    annulus(1,3) is the case — qmd.dynkin knows nothing about it.
    """
    from qmd.core import run_generation
    from qmd.dynkin import _A
    from qmd.d1_export import build_rank_rows

    tri = surfaces.triangulations_for(4, samples=60_000)
    seeds = [surfaces.quiver_of(*tri[(0, 2, 0, (1, 3))]), _A(4)]
    result = run_generation(max_vertices=4, bound=2, ranks=[4], node_cap=100, seeds=seeds)
    rows = build_rank_rows(result, 4, bound=2, node_cap=100, la_timeout=0)
    labels = {r["id"]: (r["dynkin_type"], r["label"]) for r in rows["mutation_classes"]}
    assert labels["MC.n4.a8bde37bead959e3"] == (None, "annulus(1,3)")
    assert labels["MC.n4.e6a3dea3a49e8c22"] == ("A4", "A4")
