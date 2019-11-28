
def flatten_dict_entries(in_dict, keys):
    out_list = []
    for k in keys:
        out_list.extend(in_dict[k])
    return out_list
