import numpy as np 


def calculate_states_all(prob_list):
    a = np.sum(prob_list, axis = 0)
    return a/a.sum()

def get_indv_state_argmax_all(pred_indv_dict, proba_indv_dict, time_indv_dict, shift_array, window_size, begin_time, end_time):
    time_list = []
    merged_indv_pred_list = []
    merged_indv_proba_list = []

    ptr = begin_time + 30000
    while  ptr - end_time < window_size*1/2*60000:
        if ptr > end_time:
            ptr = end_time
        prob_list = []
        pred_list = []

        for shift in shift_array:
            # check for beginning of each shift sequence
            if ptr > time_indv_dict[shift][0] - window_size*60000 and ptr <= time_indv_dict[shift][-1]:
                idx = next(idx for idx, value in enumerate(time_indv_dict[shift]) if value >= ptr)
                pred_ = pred_indv_dict[shift][idx]
                pred_list.append(pred_indv_dict[shift][idx])
                prob_list.append(proba_indv_dict[shift][idx])
        result_prob = calculate_states_all(prob_list)
        highest_idx = np.argmax([result_prob])

        merged_indv_pred_list.append([highest_idx])

        merged_indv_proba_list.append(np.array(result_prob)[highest_idx].flatten().tolist())
        time_list.append(ptr)
        if ptr == end_time:
            break
        ptr += 30000
    return merged_indv_pred_list, merged_indv_proba_list, time_list


def merge_segment_with_state_calculation_all(shift_array, df_dict, time_arr_dict, pred_dict, prob_dict, window_size):
    merged_pred_dict = {}
    merged_proba_dict = {}
    time_subj_dict = {}
    
    for k in pred_dict[0.5].keys():
        df = df_dict[k]
        
        begin_time = df.iloc[0,:]
        begin_time = begin_time.loc['onset_mil']
        end_time = df.iloc[-1,:]
        end_time = end_time.loc['offset_mil']
        
        time_indv_dict = {}
        pred_indv_dict = {}
        proba_indv_dict = {}
        
        merged_indv_pred_list = []
        merged_indv_proba_list = []

        for shift in shift_array:
            time_indv_dict[shift] = time_arr_dict[shift][k]
            pred_indv_dict[shift] = pred_dict[shift][k]
            proba_indv_dict[shift] = prob_dict[shift][k]
        
        merged_indv_pred_list, merged_indv_proba_list, time_list = get_indv_state_argmax_all(pred_indv_dict, proba_indv_dict, time_indv_dict, shift_array, window_size, begin_time, end_time)  
        
        time_subj_dict[k] = time_list
        merged_proba_dict[k] = merged_indv_proba_list
        merged_pred_dict[k] = merged_indv_pred_list
    return merged_pred_dict, merged_proba_dict, time_subj_dict

def merge_general(state_list, time_list):
    new_state_list = []
    new_time_list = []
    ptr = time_list[0]
    current_state = state_list[0]
    
    for idx in range(1, len(time_list)):
        if idx == 1 and state_list[idx] != state_list[idx - 1]:
                new_state_list.append(current_state)
                new_time_list.append(ptr)
                ptr = time_list[idx]
                current_state = state_list[idx]
        elif idx == len(time_list) - 1:
            if state_list[idx] != state_list[idx - 1]:
                new_state_list.append(current_state)
                new_time_list.append(time_list[idx-1])
                new_state_list.append(state_list[idx])
                new_time_list.append(time_list[idx])
            else:
                new_state_list.append(state_list[idx])
                new_time_list.append(time_list[idx])
        elif state_list[idx] != state_list[idx - 1]:
            new_state_list.append(current_state)
            new_time_list.append(time_list[idx-1])
            current_state = state_list[idx]
            ptr = time_list[idx-1]
    return new_state_list, new_time_list