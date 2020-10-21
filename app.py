import streamlit as st
import numpy as np 
from pathlib import Path 
from matplotlib import pyplot as plt 
import plotly.express as px 

st.title('Demo UI for Infants\' toy selection behaviors')

n_states = st.sidebar.slider("Number of states", 2, 6, 5, 1)
window_sizes = st.sidebar.slider("Window sizes", 1., 2.5, 2., 0.5)
subject_n = st.sidebar.multiselect("Subject IDs", [26, 28, 30, 31, 33, 34, 38, 48, 49, 51, 53, 58, 59, 66, 80, 25, 37, 69, 70, 84])
features_list = ["# Switch", "# toys", "# new toys", "dom toy ratio", "toy IOU"]

def main():
    emission_img = get_emission_prob(n_states, window_sizes)
    st.image(emission_img, use_column_width=True)

    all_images = get_timeline(n_states, window_sizes, subject_n)
    for img in all_images:
        st.image(img, use_column_width=True)

def get_timeline(n_states, window_sizes, subject_n):
    all_images = []
    for subject in subject_n:
        all_images.append('./figures/20200907/'+str(window_sizes)+'/'+str(int(n_states))+'/merged_gen/'+str(subject)+'.png')
    return all_images

def get_emission_prob(n_states, window_sizes):
    if window_sizes.is_integer():
        file_name = './figures/emission_prob/window_'+str(int(window_sizes))+'_n_states_'+str(int(n_states))+'.png'
    else:
        file_name = './figures/emission_prob/window_'+str(window_sizes)+'_n_states_'+str(int(n_states))+'.png'
    return file_name

if __name__ == "__main__":
    main()