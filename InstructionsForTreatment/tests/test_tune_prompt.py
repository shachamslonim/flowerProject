"""Tests for the deterministic parts of the prompt-tuning loop."""

from unittest.mock import patch

from InstructionsForTreatment.data_loader import load_flower_data
from InstructionsForTreatment.template_renderer import render_instructions_en
from InstructionsForTreatment.tune_prompt import (
    build_prompt,
    expected_facts,
    find_contradictions,
    propose_rules,
    _DEFAULT_SYSTEM_BASE_EN,
    _norm,
)


def _check(candidate, flower, is_open):
    data = load_flower_data(flower)
    tx = _norm(candidate)
    missing = [label for label, pred in expected_facts(data, is_open) if not pred(tx)]
    contra = find_contradictions(candidate, data)
    return missing, contra


class TestExpectedFacts:
    def test_template_output_has_zero_missing(self):
        """The ground-truth template must satisfy all of its own facts."""
        for flower, is_open in [("Rose", True), ("Anemone", False), ("Sunflower", False)]:
            data = load_flower_data(flower)
            truth = render_instructions_en(data, is_open, False, False)
            missing, contra = _check(truth, flower, is_open)
            assert missing == [], f"{flower}: template missing {missing}"
            assert contra == [], f"{flower}: template contradictions {contra}"

    def test_detects_missing_facts(self):
        missing, _ = _check("Put the flowers in water.", "Rose", True)
        assert any("acetone" in m for m in missing)
        assert any("20" in m for m in missing)
        assert any("50 cm" in m for m in missing)

    def test_rose_facts_include_water_height_and_rates(self):
        labels = [lbl for lbl, _ in expected_facts(load_flower_data("Rose"), True)]
        assert any("water bucket height 50 cm" in l for l in labels)
        assert any("TOG-6" in l and "rate" in l for l in labels)


class TestContradictions:
    def test_wrong_temperature(self):
        c = find_contradictions("Store at 25°C for best results.", load_flower_data("Rose"))
        assert any("25" in x for x in c)

    def test_wrong_humidity_only_near_humidity_words(self):
        data = load_flower_data("Rose")
        assert find_contradictions("Mix TOG-10 at 10% concentration.", data) == []
        assert any("80%" in x for x in find_contradictions("Keep humidity around 80%.", data))

    def test_wrong_bucket_height(self):
        c = find_contradictions("Fill the bucket to 30 cm.", load_flower_data("Rose"))
        assert any("30 cm" in x and "50 cm" in x for x in c)

    def test_correct_values_no_contradiction(self):
        data = load_flower_data("Rose")
        good = "Temperature 20°C, humidity 30%, bucket height 50 cm."
        assert find_contradictions(good, data) == []


class TestProposeRules:
    def test_parses_general_rule_lines(self):
        reply = (
            "- Include every product from the data with its exact concentration\n"
            "- Reproduce every numeric value exactly; never round or invent numbers\n"
        )
        with patch("InstructionsForTreatment.tune_prompt.ollama_chat", return_value=reply):
            rules = propose_rules("judge", ["'TOG-30' rate 0.035"], [])
        assert len(rules) == 2
        assert all(r.startswith("- ") and r.endswith(".") for r in rules)

    def test_rejects_rules_that_smuggle_in_data(self):
        reply = (
            "- Use TOG-30 at 0.035 concentration\n"           # product + number
            "- Sterilize with acetone for 20 minutes\n"        # specifics
            "- Keep the temperature at 20°C\n"                  # number + degree
            "- List every product with its exact concentration value\n"  # OK
        )
        with patch("InstructionsForTreatment.tune_prompt.ollama_chat", return_value=reply):
            rules = propose_rules("judge", [], [])
        assert rules == ["- List every product with its exact concentration value."]

    def test_judge_failure_returns_no_rules(self):
        with patch(
            "InstructionsForTreatment.tune_prompt.ollama_chat",
            side_effect=RuntimeError("judge offline"),
        ):
            assert propose_rules("judge", ["x"], []) == []


class TestBuildPrompt:
    def test_no_rules_is_the_default(self):
        assert build_prompt([]).strip() == _DEFAULT_SYSTEM_BASE_EN.strip()

    def test_rules_are_appended_in_rules_block(self):
        out = build_prompt(["- Copy every number exactly."])
        assert "- Copy every number exactly." in out
        assert out.index("Copy every number") > out.index("Rules:")
