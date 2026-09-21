"""The semantic edit tokens: what an edit did, stable across machines."""

from mapguard.features.edit_tokens import TOKENS_PER_EDIT, edit_tokens, encode_tokens, token_id


def test_tokens_describe_added_removed_and_changed_tags():
    t = edit_tokens("modify", "way", {"highway": "residential", "historic": "yes", "name": "A"},
                    {"highway": "motorway", "name": "B", "leisure": "park"})
    assert t[0] == "modify_way"
    assert "~highway:residential>motorway" in t and "~highway" in t
    assert "-historic" in t and "-historic=yes" in t
    assert "+leisure=park" in t
    assert "~name" in t and not any(x.startswith("~name:") for x in t)   # free text is not a token


def test_create_delete_and_unknown_previous_version():
    assert edit_tokens("create", "node", {}, {"amenity": "bench"}) == ["create_node", "+amenity", "+amenity=bench"]
    assert edit_tokens("delete", "node", {"shop": "bakery"}, {}) == ["delete_node", "-shop", "-shop=bakery"]
    assert edit_tokens("modify", "node", None, {"shop": "bakery"})[:2] == ["modify_node", "prev_unknown"]


def test_ids_are_stable_padded_and_never_zero():
    ids = encode_tokens("create_node +amenity")
    assert len(ids) == TOKENS_PER_EDIT and ids[2:] == [0] * (TOKENS_PER_EDIT - 2)
    assert ids[0] == token_id("create_node") and 0 not in ids[:2]
    assert token_id("+leisure=park") == token_id("+leisure=park")
