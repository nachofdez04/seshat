from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from seshat.core.models.documents import (
    DocumentKind,
    DocumentReviewRequest,
    DocumentValidationStatus,
    GeneratedDocument,
    document_is_publishable,
    effective_content,
)
from seshat.core.utils.hashing import sha256_text


def _make_document(**overrides) -> GeneratedDocument:
    markdown = overrides.pop("markdown_content", "# Meeting Summary\n")
    fields = {
        "job_id": "job-1",
        "kind": DocumentKind.MEETING_SUMMARY,
        "filename": "meeting_summary.md",
        "markdown_content": markdown,
        "content_revision": sha256_text(markdown),
        "created_at": datetime.now(UTC),
    }
    fields.update(overrides)
    return GeneratedDocument(**fields)


class TestDocumentReviewRequest:
    def test_edited_content_rejected_on_reject_action(self):
        with pytest.raises(ValidationError, match="edited_content"):
            DocumentReviewRequest(
                action="reject",
                expected_revision="abc",
                expected_validation_revision=0,
                edited_content="# Edited\n",
            )

    def test_approve_with_edited_content_is_valid(self):
        request = DocumentReviewRequest(
            action="approve",
            expected_revision="abc",
            expected_validation_revision=0,
            edited_content="# Edited\n",
        )
        assert request.edited_content == "# Edited\n"

    def test_reject_with_reason_is_valid(self):
        request = DocumentReviewRequest(
            action="reject",
            expected_revision="abc",
            expected_validation_revision=0,
            reason="wrong decisions",
        )
        assert request.reason == "wrong decisions"


def test_effective_content_returns_original_without_edit():
    doc = _make_document()
    assert effective_content(doc) == doc.markdown_content


def test_effective_content_prefers_edited_content():
    doc = _make_document(edited_content="# Edited\n")
    assert effective_content(doc) == "# Edited\n"


def test_document_is_publishable_false_for_pending_and_rejected():
    assert not document_is_publishable(_make_document())
    assert not document_is_publishable(
        _make_document(validation_status=DocumentValidationStatus.REJECTED, rejection_reason="no")
    )


def test_document_is_publishable_true_for_matching_approval():
    doc = _make_document(
        validation_status=DocumentValidationStatus.APPROVED,
        approved_revision=sha256_text("# Meeting Summary\n"),
    )
    assert document_is_publishable(doc)


def test_document_is_publishable_true_for_edited_with_matching_hash():
    doc = _make_document(
        validation_status=DocumentValidationStatus.EDITED,
        edited_content="# Edited\n",
        approved_revision=sha256_text("# Edited\n"),
    )
    assert document_is_publishable(doc)


def test_document_is_publishable_false_for_tampered_content():
    doc = _make_document(
        validation_status=DocumentValidationStatus.APPROVED,
        approved_revision=sha256_text("# Something else entirely\n"),
    )
    assert not document_is_publishable(doc)


def test_document_is_publishable_false_for_stale_source_revision_after_edit():
    doc = _make_document(
        markdown_content="# Regenerated source\n",
        content_revision=sha256_text("# Original source\n"),
        validation_status=DocumentValidationStatus.EDITED,
        edited_content="# Approved edit\n",
        approved_revision=sha256_text("# Approved edit\n"),
    )
    assert not document_is_publishable(doc)


def test_document_is_publishable_false_without_approved_revision():
    doc = _make_document(validation_status=DocumentValidationStatus.APPROVED)
    assert not document_is_publishable(doc)
