"""String helpers used by the sample project."""


def slugify(text):
    return text.lower().strip().replace(" ", "-")
