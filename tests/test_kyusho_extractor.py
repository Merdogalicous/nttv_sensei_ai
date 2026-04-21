import pathlib

from extractors.kyusho import try_answer_kyusho

KYU = pathlib.Path("data") / "KYUSHO.txt"


def _passages():
    return [
        {
            "text": KYU.read_text(encoding="utf-8"),
            "source": "KYUSHO.txt",
            "meta": {"priority": 1},
        }
    ]


def test_specific_point_ura_kimon_from_real_data():
    ans = try_answer_kyusho("Where is the Ura Kimon kyusho point?", _passages())
    assert ans and ans.answer_type == "kyusho_point"
    assert ans.facts["point_name"].lower() == "ura kimon"
    assert "ribs under the pectoral muscles" in ans.facts["description"].lower()
