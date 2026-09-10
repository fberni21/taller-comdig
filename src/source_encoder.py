def compute_text_stats(text):
    return text


def compute_file_stats(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return compute_text_stats(text)


class SourceEncoder:
    pass
