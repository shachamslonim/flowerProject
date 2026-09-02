"""Tests for Hebrew template renderer."""

import pytest

from InstructionsForTreatment.data_loader import load_flower_data
from InstructionsForTreatment.template_renderer import render_instructions_he


class TestHebrewRenderer:
    """Tests for render_instructions_he function."""

    def test_rose_header_includes_hebrew_name(self):
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=True, yellow_leaves=False, leaves_falling=False)
        assert "ורד" in result

    def test_rose_open_includes_water_entry_time_hebrew(self):
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=True, yellow_leaves=False, leaves_falling=False)
        assert "15 דקות" in result

    def test_rose_open_includes_water_height_50(self):
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=True, yellow_leaves=False, leaves_falling=False)
        assert '50 ס"מ' in result

    def test_rose_closed_shows_controlled_harvest(self):
        # is_open=True renders the CLOSED-flower handling (flag meaning is inverted)
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=True, yellow_leaves=False, leaves_falling=False)
        assert "קטיף מבוקר" in result
        assert "סגורים" in result

    def test_sterilization_section_includes_acetone_hebrew(self):
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "אצטון" in result

    def test_sterilization_includes_all_tools_hebrew(self):
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "מיכלים" in result
        assert "מזמרות" in result
        assert "שולחנות" in result
        assert "גיליוטינה" in result

    def test_anemone_high_sensitivity_includes_sts(self):
        """Anemone has ethyleneSensitivity=3, should recommend STS in Hebrew."""
        data = load_flower_data("Anemone")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "STS" in result
        assert "TOG-L-101" in result

    def test_sts_explains_ethylene_sensitivity_hebrew(self):
        """STS section should explain ethylene blocking in Hebrew."""
        data = load_flower_data("Anemone")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "אתילן" in result
        assert "נשירה" in result

    def test_rose_low_sensitivity_no_sts(self):
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "TOG-L-101" not in result

    def test_yellow_leaves_includes_gibberellin_hebrew(self):
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=False, yellow_leaves=True, leaves_falling=False)
        assert "ג'יברלין" in result
        assert "הצהבת" in result

    def test_leaves_falling_includes_101L_hebrew(self):
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=True)
        assert "101L" in result
        assert "נשירת עלים" in result

    def test_includes_temperature_and_moisture_hebrew(self):
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "20°C" in result
        assert "30%" in result

    def test_includes_farmer_treatments(self):
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "TOG-6" in result
        assert "ריכוז" in result

    def test_treatments_show_rank_order_hebrew(self):
        """Treatments should show Hebrew rank labels."""
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "אפשרות 1" in result
        assert "אפשרות 2" in result

    def test_treatments_sorted_by_priority_hebrew(self):
        """TOG-6 should appear before TOG-10 in Hebrew output."""
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        pos_tog6 = result.index("TOG-6")
        pos_tog10 = result.index("TOG-10")
        assert pos_tog6 < pos_tog10

    def test_recommended_treatment_is_tagged_hebrew(self):
        """Products flagged recommended in flowers.json are marked in Hebrew."""
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "מומלץ" in result

    def test_multiple_recommended_lead_the_list_hebrew(self):
        """Aconitum has two recommended products; both come before the standard ones."""
        data = load_flower_data("Aconitum")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert result.index("TOG-30") < result.index("TOG-6")
        assert result.index("TOG-75") < result.index("TOG-6")

    def test_materials_explains_biocides_hebrew(self):
        """Materials section should explain these are biocides in Hebrew."""
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "ביוצידים" in result
        assert "פטריות" in result
        assert "חיידקים" in result

    def test_null_water_height_shows_not_specified_hebrew(self):
        data = load_flower_data("Anemone")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "לא צוין" in result

    def test_anemone_header_includes_hebrew_name(self):
        data = load_flower_data("Anemone")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "אדמונית" in result

    def test_open_flower_explains_timing_critical_hebrew(self):
        """Open flower (is_open=False) should explain timing is critical in Hebrew."""
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "קריטי" in result

    def test_water_entry_section_mentions_sink(self):
        """Closed-flower instruction (is_open=True) mentions sink."""
        data = load_flower_data("Rose")
        result = render_instructions_he(data, is_open=True, yellow_leaves=False, leaves_falling=False)
        assert "sink" in result
