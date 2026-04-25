"""Unit tests for the RAG pipeline components."""

import pytest
from unittest.mock import patch, MagicMock


class TestIntentClassifier:
    """Test intent classification and escalation."""

    def test_escalation_keyword_triggers_hitl(self):
        """Verify escalation keywords trigger human escalation."""
        from src.agents.rag_graph import intent_classifier_node

        state = {
            "query": "I want to sue your company for this issue",
            "escalated": False,
            "escalation_reason": "",
            "retrieved_docs": [],
            "relevant_docs": [],
            "answer": "",
            "confidence": 1.0,
            "session_id": "test_001",
            "messages": []
        }
        config = {
            "configurable": {
                "escalation_keywords": ["sue", "legal", "fraud"]
            }
        }

        result = intent_classifier_node(state, config)
        assert result["escalated"] is True
        assert "sue" in result["escalation_reason"]

    def test_normal_query_passes_through(self):
        """Verify normal queries proceed to RAG pipeline."""
        from src.agents.rag_graph import intent_classifier_node

        state = {
            "query": "What is your return policy?",
            "escalated": False,
            "escalation_reason": "",
            "retrieved_docs": [],
            "relevant_docs": [],
            "answer": "",
            "confidence": 1.0,
            "session_id": "test_002",
            "messages": []
        }
        config = {"configurable": {"escalation_keywords": ["sue", "legal", "fraud"]}}

        result = intent_classifier_node(state, config)
        assert result["escalated"] is False


class TestRouting:
    """Test routing logic between graph nodes."""

    def test_route_after_grader_with_no_docs(self):
        """Route to HITL when no relevant docs are found."""
        from src.agents.rag_graph import route_after_grader

        state = {"relevant_docs": [], "confidence": 0.8}
        assert route_after_grader(state) == "hitl"

    def test_route_after_grader_with_docs(self):
        """Route to generator when relevant docs exist."""
        from src.agents.rag_graph import route_after_grader

        state = {"relevant_docs": ["Some relevant context here."], "confidence": 0.8}
        assert route_after_grader(state) == "generator"

    def test_route_after_generator_low_confidence(self):
        """Route to HITL when confidence is below threshold."""
        from src.agents.rag_graph import route_after_generator

        state = {"confidence": 0.3}
        config = {"configurable": {"confidence_threshold": 0.60}}
        assert route_after_generator(state, config) == "hitl"

    def test_route_after_generator_high_confidence(self):
        """End pipeline when confidence is above threshold."""
        from src.agents.rag_graph import route_after_generator

        state = {"confidence": 0.9}
        config = {"configurable": {"confidence_threshold": 0.60}}
        assert route_after_generator(state, config) == "end"


class TestDocumentLoader:
    """Test document loading and chunking."""

    def test_chunk_metadata_populated(self):
        """Verify chunk metadata includes chunk_id and source."""
        from src.utils.document_loader import load_and_chunk_pdf

        mock_page = MagicMock()
        mock_page.page_content = "This is a test document about refunds. " * 20
        mock_page.metadata = {"page": 0}

        with patch("src.utils.document_loader.PyPDFLoader") as mock_loader:
            mock_loader.return_value.load.return_value = [mock_page]
            chunks = load_and_chunk_pdf("dummy.pdf", chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 0
        assert all("chunk_id" in c.metadata for c in chunks)
        assert all("source" in c.metadata for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
