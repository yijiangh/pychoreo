
def is_any_empty(in_list):
    """A convinient function to check if all the entry in a list is not None.
    """
    if in_list == []:
        return True
    else:
        return any((isinstance(sli, list) and is_any_empty(sli)) for sli in in_list)
