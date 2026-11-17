from types import SimpleNamespace

from capt_pronunciation.contracts import CaptSettings
from capt_pronunciation.diagnostics import Wav2Vec2PhonemeRecognizer


class _FakePhoneTokenizer:
    unk_token_id = 3

    def phonemize(self, word: str) -> str:
        assert word == "course"
        return "k ɔːɹ s |"

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"k": 11, "ɔːɹ": 71, "s": 5}.get(token, self.unk_token_id)


def test_canonical_phones_keep_model_inventory_tokens():
    recognizer = Wav2Vec2PhonemeRecognizer(CaptSettings())
    recognizer.processor = SimpleNamespace(tokenizer=_FakePhoneTokenizer())
    recognizer.model = object()

    assert recognizer.canonical_phones("course", ["k", "ɔ", "r", "s"]) == [
        "k",
        "ɔːɹ",
        "s",
    ]
