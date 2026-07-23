import pytest

from seshat.eval.models import GateResult
from tests.integration.conftest import SKIP_IF_NO_ASSEMBLYAI_API
from tests.integration.eval.helpers import make_eval_config, make_transcription_runner

pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.transcription,
    pytest.mark.eval,
    SKIP_IF_NO_ASSEMBLYAI_API,
]


class TestTranscriptionEvalRunner:
    async def test_run_produces_gate_result_and_file(self, tmp_path):
        """Live smoke test against the real provider — asserts shape, not accuracy.

        The test corpus reference is a placeholder (see data/eval/test_corpora/transcription),
        so the WER value here is meaningless; WER correctness is covered by the scorer unit tests.
        """
        config = make_eval_config(tmp_path)
        runner = make_transcription_runner(config)
        result = await runner.run()

        assert isinstance(result, GateResult)
        assert result.run_id
        assert result.transcription_metrics is not None
        assert "wer" in result.transcription_metrics

        assert (tmp_path / "eval_gate.json").exists()
