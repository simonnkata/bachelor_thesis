from DataLoader.dataloader_helpers import load_data, apply_pos, filter_data, normalise, Subjects_nd


def load_process_data() -> Subjects_nd:
    original_data = load_data()
    processed_data = apply_pos(original_data)
    filter_data(processed_data)
    normalise(processed_data)
    return processed_data
