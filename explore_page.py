import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pydeck as pdk



@st.cache_data
def load_data():
   df=pd.read_csv('mumbai-house-price-data-cleaned.csv')
   return df

df=load_data()

def show_explore_page():
   st.title("EXPLORE")
   st.subheader("Price Distribution")
   df_geo = df[
    (df['latitude'].between(18.5, 19.8)) &
    (df['longitude'].between(72.5, 73.5))
     ].copy()

   fig1,ax1 = plt.subplots()
   st.subheader("Spatial Distribution (Price per Sqft)")

   layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_geo,
    get_position='[longitude, latitude]',
    get_radius=25,               
    get_fill_color='[255, (1 - log_ppsqft / 6) * 255, 0]',  
    pickable=True,
    opacity=0.6
    )

   view_state = pdk.ViewState(
    latitude=19.0760,
    longitude=72.8777,
    zoom=10,
    pitch=0
     )

   deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "text": "₹/sqft: {price_per_sqft}"
    },
    
)
   
   st.pydeck_chart(deck)
   ax1.hist(df['price_per_sqft'],bins=50)
   ax1.set_yscale('log')
   ax1.set_xlabel('Price per sqft(log scale)')
   ax1.set_ylabel("count")
   st.pyplot(fig1)

   bhk = st.selectbox("Select BHK", sorted(df['bedroom_num'].unique()))
   temp = df[df['bedroom_num'] == bhk]

   fig, ax = plt.subplots()
   ax.scatter(temp['area'], temp['price'], alpha=0.3)
   ax.set_xlabel("Area (sqft)")
   ax.set_ylabel("Price")
   st.pyplot(fig)

   
   
   

   


   



   

 

