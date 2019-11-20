
def is_any_empty(sols):
    """A convinient function to check if all the entry in a list is not None.
    """
    return not sols or any(not val for val in sols)
