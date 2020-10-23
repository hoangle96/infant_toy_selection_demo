from matplotlib.pyplot import plot
import streamlit as st
import numpy as np 
from pathlib import Path 
import hmm_model as hmm 
import pickle
from merge import merge_segment_with_state_calculation_all, merge_general
from draw_functions import draw_distribution, draw_merged_timeline
from itertools import chain

st.title('Demo UI for Infants\' toy selection behaviors')

subjects_list = [26, 28, 30, 31, 33, 34, 38, 48, 49, 51, 53, 58, 59, 66, 80, 25, 37, 69, 70, 84]
n_states = st.sidebar.slider("Number of states", 2, 6, 5, 1)
window_sizes = st.sidebar.slider("Window sizes", 1., 2.5, 2., 0.5)
subject_n = st.sidebar.multiselect("Subject IDs", subjects_list, default=[80])
features_name_list = ["# toy switches", "# toys", "# new toys", "fav toy ratio", "toy IOU"]
feature_list_multiselect = st.sidebar.multiselect('features list', features_name_list, default=features_name_list)
shift_time_dict = {1:{0, 0.5}, 1.5:{0, 0.5, 1}, 2:[0,0.5,1,1.5], 2.5:[0,0.5,1,1.5, 2], 3:[0,0.5,1,1.5, 2, 2.5]}

def main():
    global window_sizes
    with open(Path('./data/annotated_episode_dict_20200904.pickle'), 'rb') as f:
        merged_data_dict = pickle.load(f) 
    window_sizes = check_window_size_int(window_sizes)
    features_list, feature_vector, feature_dict = get_features(feature_list_multiselect, features_name_list, window_sizes)
    model = fit_model(n_states, features_list, feature_vector)
    state_merged_general, time_merged_general, pred_list = get_prediction(model, features_list, subjects_list, window_sizes, shift_time_dict, merged_data_dict)
    
    labels = get_features_x_labels(feature_dict, window_sizes)
    fig, _ = draw_distribution(n_states, feature_vector, feature_list_multiselect, np.array(pred_list), labels)
    st.write("Emission distribution for model with window size of "+ str(window_sizes)+" minute(s), " + str(n_states) + " states, and " + str(len(feature_list_multiselect)) + " features")
    st.pyplot(fig)

    st.write("Individual state trajectories")

    plotly_list = get_subject_timeline(subject_n, merged_data_dict, state_merged_general, time_merged_general)
    for plot in plotly_list:
        st.plotly_chart(plot)

# @st.cache()
def get_features(feature_list_multiselect, features_name_list, window_sizes):
    ''' feature list is a list of individual (timestep, n_features) np.arrays. Use this to fit hmm.
        feature vector is the aggregated of (all, n_features) np.array. Use this to instantiate hmm.
    '''
    selected_features = feature_list_multiselect
    if len(selected_features) == 0:
        selected_features = feature_list_multiselect

    idx_selected_features = []
    for i in range(len(selected_features)):
        idx_selected_features.append(features_name_list.index(selected_features[i]))

    with open('./data/feature_list_window_'+str(window_sizes)+'.pickle', 'rb') as f:
        feature_list = pickle.load(f)

    new_features_list = []
    for indv_feature_list in feature_list:
        new_features_list.append(indv_feature_list[:, idx_selected_features])
    
    with open('./data/feature_window_'+str(window_sizes)+'.pickle', 'rb') as f:
        feature_vector= pickle.load(f)
    
    feature_dict = {}
    feature_vector = np.array(feature_vector)[:, idx_selected_features]
    for i in range(len(feature_vector.T)):
        feature_dict[feature_list_multiselect[i]] = feature_vector.T[i]

    return new_features_list, feature_vector, feature_dict

@st.cache(show_spinner=False)
def fit_model(n_states, features_list, feature_vector):
    model = hmm.init_hmm(n_components = n_states, feature_list = feature_vector.T, seed = 0)
    model.bake()
    model.fit(features_list)
    return model

@st.cache(show_spinner=False)
def load_original_dataset():
    with open(Path('./data/annotated_episode_dict_20200904.pickle'), 'rb') as f:
        return pickle.load(f) 
    
@st.cache(show_spinner=False)
def load_time_arr_dict(window_size):
    with open(Path('./data/time_arr_dict_'+str(window_size)+'.pickle'), 'rb') as f:
        return pickle.load(f) 

def check_window_size_int(window_size):
    if not isinstance(window_size, int):
        window_size = int(window_size) if window_size.is_integer() else window_size
    return window_size

@st.cache(show_spinner=False)
def get_prediction(model, list_seq, subjects_list, window_size, shift_time_dict, merged_df_dict):
    pred_list = []
    proba_list = []
    for seq in list_seq:
        pred_list.append(model.predict(seq))
        proba_list.append(model.predict_proba(seq))

    pred_subj_dict = {}
    prob_subj_dict = {}
    discretized_feature_subj_dict = {}
    # shift_time_dict = load_time_arr_dict(window_size)
    i = 0
    for shift in shift_time_dict[window_size]:
        pred_each_shift = {}
        prob_each_shift = {}
        discretized_feature_each_shift = {}

        for k in subjects_list:
            pred_each_shift[k] = pred_list[i]
            prob_each_shift[k] = proba_list[i]
            discretized_feature_each_shift[k] = list_seq[i]
            i += 1
        pred_subj_dict[shift] = pred_each_shift
        prob_subj_dict[shift] = prob_each_shift
        discretized_feature_subj_dict[shift] = discretized_feature_each_shift
    
    # merged_df_dict = load_original_dataset()

    window_size = check_window_size_int(window_size)
    time_arr_shift_dict = load_time_arr_dict(window_size)

    merged_pred_dict_all, merged_proba_dict_all, time_subj_dict_all = merge_segment_with_state_calculation_all(shift_time_dict[window_size], merged_df_dict, time_arr_shift_dict, pred_subj_dict, prob_subj_dict, window_size)    
    
    state_merged_general = {}
    prod_merged_general = {}
    time_merged_general = {}
    for k in merged_pred_dict_all.keys():
        state_list = np.array(merged_pred_dict_all[k]).flatten().tolist()
        prob_list = merged_proba_dict_all[k]
        time_list = time_subj_dict_all[k]
        new_state_list, new_time_list = merge_general(state_list, time_list)
        state_merged_general[k] = new_state_list
        time_merged_general[k] = new_time_list  
    
    return state_merged_general, time_merged_general, list(chain(*pred_list))

def get_features_x_labels(feature_dict, window_size):
    labels = []
    ep_rate_dict = {1:[0, 2, 4, 6, 8], 1.5: [0, 3, 6, 9, 12], 2:[0, 3, 6, 9, 12], 2.5: [0, 4, 8, 12, 16], 3: [0, 5, 10, 15, 20]}

    for f_i in feature_dict.keys():
        if "# toy switches" == f_i:
            labels.append(ep_rate_dict[window_size])
        elif "# toys" == f_i:
            labels.append(np.unique(feature_dict[f_i]))
        elif "# new toys" == f_i:
            labels.append(np.unique(feature_dict[f_i]))
        elif "fav toy ratio" == f_i:
            labels.append([0, .2, .4, .6, .8])
        elif "toy IOU" == f_i:
            labels.append([0, .2, .4, .6, .8])
    return labels 

def get_subject_timeline(subj_list, merged_df_dict, merged_pred_dict_all, time_subj_dict_all):
    plotly_list = []
    for k in subj_list:
        state_list = merged_pred_dict_all[k]
        time_list = time_subj_dict_all[k]
        # st.write(time_list)
        plotly_list.append(draw_merged_timeline(k, merged_df_dict, state_list, time_list))
    return plotly_list


#  def get_timeline(n_states, window_sizes, subject_n):
#     all_images = []
#     for subject in subject_n:
#         all_images.append('./figures/20200907/'+str(window_sizes)+'/'+str(int(n_states))+'/merged_gen/'+str(subject)+'.png')
#     return all_images

# def get_emission_prob(n_states, window_sizes):
#     if window_sizes.is_integer():
#         file_name = './figures/emission_prob/window_'+str(int(window_sizes))+'_n_states_'+str(int(n_states))+'.png'
#     else:
#         file_name = './figures/emission_prob/window_'+str(window_sizes)+'_n_states_'+str(int(n_states))+'.png'
#     return file_name

if __name__ == "__main__":
    main()