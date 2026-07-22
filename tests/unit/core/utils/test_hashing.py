from __future__ import annotations

import hashlib

from seshat.core.utils.hashing import sha256_text


def test_sha256_text_matches_utf8_digest():
    text = "Reunión de equipo — decisión final"
    assert sha256_text(text) == hashlib.sha256(text.encode()).hexdigest()


def test_sha256_text_is_stable_across_calls():
    assert sha256_text("same input") == sha256_text("same input")


def test_sha256_text_differs_for_different_input():
    assert sha256_text("a") != sha256_text("b")
