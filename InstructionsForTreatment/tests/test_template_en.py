"""Tests for English template renderer."""

import pytest

from InstructionsForTreatment.data_loader import load_flower_data
from InstructionsForTreatment.template_renderer import render_instructions_en


class TestEnglishRenderer:
    """Tests for render_instructions_en function."""

    def test_rose_open_includes_water_entry_time(self):
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=True, yellow_leaves=False, leaves_falling=False)
        assert "15 minutes" in result

    def test_rose_open_includes_water_height_50(self):
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=True, yellow_leaves=False, leaves_falling=False)
        assert "50 cm" in result

    def test_rose_closed_shows_controlled_harvest(self):
        # is_open=True renders the CLOSED-flower handling (flag meaning is inverted)
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=True, yellow_leaves=False, leaves_falling=False)
        assert "Controlled harvest" in result
        assert "15 minutes" in result

    def test_anemone_high_sensitivity_includes_sts(self):
        """Anemone has ethyleneSensitivity=3, should recommend STS."""
        data = load_flower_data("Anemone")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "STS" in result
        assert "TOG-L-101" in result

    def test_sts_explains_ethylene_blocking(self):
        """STS section should explain it prevents petal drop."""
        data = load_flower_data("Anemone")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "petal drop" in result
        assert "ethylene" in result.lower()

    def test_rose_low_sensitivity_no_sts(self):
        """Rose has ethyleneSensitivity='1 (2)', should NOT recommend STS."""
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "TOG-L-101" not in result

    def test_yellow_leaves_includes_gibberellin(self):
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=False, yellow_leaves=True, leaves_falling=False)
        assert "Gibberellin" in result
        assert "yellowing" in result

    def test_leaves_falling_includes_101L(self):
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=True)
        assert "101L" in result
        assert "leaf drop" in result

    def test_no_yellow_leaves_no_gibberellin(self):
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "Gibberellin" not in result

    def test_includes_sterilization_section(self):
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "acetone" in result
        assert "bins" in result
        assert "shears" in result
        assert "tables" in result
        assert "guillotine" in result

    def test_includes_temperature_and_moisture(self):
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "20\u00b0C" in result
        assert "30%" in result

    def test_rose_includes_farmer_treatments(self):
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "TOG-6" in result
        assert "0.0089" in result

    def test_treatments_show_rank_order(self):
        """Treatments should show rank labels (Option 1, Option 2, etc.)."""
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "Option 1" in result
        assert "Option 2" in result

    def test_treatments_sorted_by_priority(self):
        """Within one priority group, TOG-6 < TOG-10 < TOG-30 (tie-breaker order)."""
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        pos_tog6 = result.index("TOG-6")
        pos_tog10 = result.index("TOG-10")
        pos_tog30 = result.index("TOG-30")
        assert pos_tog6 < pos_tog10 < pos_tog30

    def test_recommended_treatment_is_tagged(self):
        """Products flagged recommended in flowers.json are labelled as such."""
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "recommended" in result

    def test_recommended_priority_outranks_product_order(self):
        """Aconitum flags TOG-30 and TOG-75 as recommended, so they lead the list.

        This must hold even though TOG-6 comes first in PRODUCT_PRIORITY_ORDER -
        the JSON `priority` field wins.
        """
        data = load_flower_data("Aconitum")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "TOG-30 (Option 1, recommended)" in result
        assert "TOG-75 (Option 2, recommended)" in result
        assert result.index("TOG-30") < result.index("TOG-6")

    def test_standard_treatments_not_tagged_recommended(self):
        """Ageratum's TOG-6 is standard, so its line carries no recommended tag."""
        data = load_flower_data("Ageratum")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        tog6_line = next(ln for ln in result.splitlines() if "TOG-6 " in ln)
        assert "recommended" not in tog6_line

    def test_materials_explains_biocides_purpose(self):
        """Materials section should explain these are biocides."""
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "Biocides" in result
        assert "fungi" in result.lower()
        assert "bacteria" in result.lower()

    def test_includes_header_with_flower_name(self):
        data = load_flower_data("Sunflower")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "Sunflower" in result

    def test_flower_with_null_additives_no_additives_line(self):
        """Rose has additives=null, should not show additives."""
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "Additives:" not in result

    def test_flower_with_additives_shows_additives(self):
        """Anemone has additives='coltar 0.1'."""
        data = load_flower_data("Anemone")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "coltar 0.1" in result

    def test_null_water_height_shows_not_specified(self):
        """Anemone has waterBucketHeight=null."""
        data = load_flower_data("Anemone")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "not specified" in result

    def test_flower_with_no_farmer_treatment(self):
        """Tulip has empty farmerTreatment list."""
        data = load_flower_data("Tulip")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "No farmer treatment specified" in result

    def test_flower_with_null_sensitivity(self):
        """Tulip has ethyleneSensitivity=null, should not crash or show STS."""
        data = load_flower_data("Tulip")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "TOG-L-101" not in result

    def test_open_flower_explains_timing_critical(self):
        """Open flower should explain timing is critical (is_open=False renders OPEN)."""
        data = load_flower_data("Rose")
        result = render_instructions_en(data, is_open=False, yellow_leaves=False, leaves_falling=False)
        assert "critical" in result.lower()
