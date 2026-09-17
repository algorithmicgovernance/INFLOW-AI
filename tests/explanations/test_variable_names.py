from explanations.variable_names import human_readable_variable_name


def test_viirs_lag_is_expressed_as_elapsed_time():
    assert (
        human_readable_variable_name("CHIRPS_lag_12", "viirs")
        == "Rainfall over Lake Victoria (CHIRPS) -6 months"
    )
    assert human_readable_variable_name("TAMSAT_lag_6", "viirs") == (
        "Rainfall over Lake Victoria (CHIRPS) -3 months"
    )


def test_region_codes_are_expanded_before_lag_label():
    assert human_readable_variable_name("rainfall_eeq_lag_2", "viirs") == (
        "Rainfall in Eastern Equatoria -1 month"
    )


def test_unmapped_names_are_still_readable():
    assert human_readable_variable_name("day_of_year", "viirs") == "Day of year"
