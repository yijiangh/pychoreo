
################################################
# result postprocessing utils

def divide_list_chunks(list, size_list):
    assert(sum(size_list) >= len(list))
    if sum(size_list) < len(list):
        size_list.append(len(list) - sum(size_list))
    for j in range(len(size_list)):
        cur_id = sum(size_list[0:j])
        yield list[cur_id:cur_id+size_list[j]]

# temp... really ugly...
def divide_nested_list_chunks(list, size_lists):
    # assert(sum(size_list) >= len(list))
    cur_id = 0
    output_list = []
    for sl in size_lists:
        sub_list = {}
        for proc_name, jt_num in sl.items():
            sub_list[proc_name] = list[cur_id:cur_id+jt_num]
            cur_id += jt_num
        output_list.append(sub_list)
    return output_list
