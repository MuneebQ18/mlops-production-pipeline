def test_schema_change_detection():
    previous_schema = {"feature_0", "feature_1", "label"}
    current_schema = {"feature_0", "feature_1", "feature_2", "label"}

    added_columns = current_schema - previous_schema
    removed_columns = previous_schema - current_schema

    assert added_columns == {"feature_2"}
    assert removed_columns == set()


def test_schema_feature_removed():
    previous_schema = {"feature_0", "feature_1", "label"}
    current_schema = {"feature_0", "label"}

    added_columns = current_schema - previous_schema
    removed_columns = previous_schema - current_schema

    assert added_columns == set()
    assert removed_columns == {"feature_1"}