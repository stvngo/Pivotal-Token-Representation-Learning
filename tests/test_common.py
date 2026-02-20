from probe_pipeline.common import parse_layer_selection


def test_parse_layer_selection_prefers_explicit():
    available = [0, 1, 2, 3, 4]
    selected = parse_layer_selection(available_layers=available, explicit_layers=[3, 1, 8], num_layers=2)
    assert selected == [3, 1]


def test_parse_layer_selection_num_layers():
    available = [5, 1, 9, 2]
    selected = parse_layer_selection(available_layers=available, explicit_layers=None, num_layers=2)
    assert selected == [1, 2]

