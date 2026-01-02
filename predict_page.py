import streamlit as st
import pickle
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import os
import lightgbm as lgb

def load_model():
    with open("mumbai2.pkl", "rb") as f:
        model = pickle.load(f)
    return model

data=load_model()

def show_predict_page():
    st.title("Mumbai House Price Prediction")

    st.write("""### We need some information to predict house price""" )

    locality=['Kalyan', 'Dombivali', 'Ville Parle', 'Nala Sopara', 'Ulwe',
       'Jogeshwari', 'Chembur', 'Nerul', 'Mulund', 'Nallasopara W',
       'Dombivli', 'Taloje', 'Panvel', 'Shil Phata', 'Vangani',
       'Kharghar', 'Nalasopara', 'Badlapur', 'Mumbra', 'Vasai',
       'Ambernath', 'Powai', 'Koper Khairane', 'Kalamboli', 'Andheri',
       'Goregaon', 'Borivali', 'Kasaradavali Thane', 'Vile Parle',
       'Kandivali', 'Mira Road', 'Malad', 'Prabhadevi', 'Dadar',
       'Lower Parel', 'Taloja', 'Wadala', 'Virar', 'Thane', 'Bhandup',
       'Nahur', 'Khar', 'Deonar', 'Ghansoli', 'Thakurli', 'Bandra',
       'Vashi', 'Bhayandar', 'Kalwa', 'Ghatkopar', 'Vikhroli', 'Mahim',
       'Parel', 'Dahisar', 'Kurla', 'Santacruz', 'Worli', 'Byculla',
       'Airoli', 'Kamothe', 'Sion', 'Juhu', 'Bhiwandi', 'Vikroli',
       'Naigaon', 'Palghar', 'Saphale', 'Makane Kapase', 'Vevoor',
       'Amboli', 'Titwala', 'Police Colony', 'Sanpada',
       'Hiranandani Estates', 'Kanjurmarg', 'Seawoods', 'Shelu',
       'Rasayani', 'Karjat', 'Tardeo', 'Nere', 'Mazagaon', 'Matunga',
       'Karanjade', 'Muthaval', 'Neral', 'Mahalaxmi', 'Shilphata',
       'Sewri', 'Juinagar', 'Navghar', 'Antop Hill', 'Saki Naka',
       'Dronagiri', 'Kolshet', 'Boisar', 'Belapur', 'Agasan Village',
       'Ibhadpada', 'Agripada', 'Mazgaon', 'Sector 19 Kharghar',
       'Usarghar', 'Anjurdive', 'Greater Khanda', 'Murbad', 'Gamdevi',
       'Malabar Hill', 'Cumballa Hill', 'Cuffe Parade',
       'Sector-12 Kamothe', 'Jambrung', 'Girgaon', 'Jacob Circle',
       'Sector 21 Kamothe', 'Sahkar Nagar', 'Fort', 'Sector 17 Ulwe',
       'Kharodi', 'Chinchpokli', 'Vichumbe', 'Ambivali', 'Khopoli',
       'Bopele', 'Mankhurd', 'Mumbai Central', 'Colaba', 'Govandi',
       'Sector 19 Ulwe', 'Patil Nagar Vasind', 'Rabale', 'Marine Lines',
       'Sector-50 Seawoods', 'Palidevad', 'Diva', 'Sector 22 Kamothe',
       'Shivkar', 'Anushakti Nagar', 'Bandra Kurla Complex', 'Nilje Gaon',
       'Sector 11 Koparkhairane', 'Chinchodyacha Pada', 'Shivaji Park',
       'Chedda Nagar', 'Sector-20 Koparkhairane', 'Madh', 'Pisarve',
       'Ambivli', 'Naupada', 'Tilak Nagar', 'Sai Samarth Mitra',
       'Majiwada', 'Sector 16 A', 'Ashok Nagar', 'Arya Nagar', 'Asangaon',
       'Shahapur', 'Sector-9 Ulwe', 'Koproli', 'Dighe', 'Pakhadi',
       'Peddar Road', 'Diva Gaon', 'Antarli', 'Sector 16 Vashi', 'Kewale',
       'Damat', 'Kiravli Village', 'Asalpha', 'Sector-9A Vashi', 'Jawhar',
       'Panch Pakhdi', 'Khardi', 'Uttan', 'Sector 17 Vashi', 'Umroli',
       'Rohinjan', 'Kongaon', 'Sector 19 Kamothe', 'Grant Road', 'Manor',
       'Pali Hill', 'Sector-14 Koparkhairane', 'Kamathipura',
       'Khadakpada', 'Kolshet Road Thane', 'Mount Marry', 'Sarsole',
       'Napeansea Road', 'Beturkar Pada', 'Additional M.I.D.C', 'Manpada',
       'Kolshet Road', 'Haranwali', 'Murarbaug', 'Louis Wadi',
       'Ghodbunder Thane', 'Pokhran Road No 2', 'Ambarnath',
       'Panch Pakhadi', 'Vasant Vihar Thane', 'Patlipada', 'Pokhran 2',
       'Ulhasnagar', 'Sector 21 Ghansoli', 'Sector15 Ghansoli',
       'Sector 5 Ghansoli', 'Koparkhairane Station Road',
       'Sector20 Koparkhairane', 'Sector-3 Ghansoli',
       'Hiranandani Meadows', 'Kalbadevi', 'Jambli Naka', 'Vile Parle E',
       'Sector 12 Kharghar', 'Parksite Colony', 'Malad Mithchowki',
       'Ghodbunder Road', 'Marine Drive', 'Churchgate', 'Gawand Baug',
       'New Panvel Navi Mumbai', 'Taloja Panvel', 'Bdd Chawls Worli',
       'Navgharh', 'Khalapur', 'Shahpur', 'Shilgaon',
       'Navpada Vile Parle', 'Mulund Gavanpada', 'Sabe Gaon', 'Vasind',
       'Parel Village', 'Balkum', 'Lbs Marg Mulund',
       'Savitribai Phule Nagar', 'Ulhasnagar 4', 'Irani Wadi',
       'Perry Cross Rd', 'Chikan Ghar', 'Jawahar Nagar', 'Lonavala',
       'Dharavi', 'Kalyan Shilphata Road', 'Navade', 'Cbd Belapur',
       'Sagaon', 'Sector-15 Ghansoli', 'Kanjur Marg',
       'Haji Malang Road Kalyan E', 'Ern Express Highway Vikhroli',
       'Hendre Pada', 'Karanjade Panvel', 'Vikramgad', 'Sector 23 Ulwe',
       'Devad', 'Ghansoli Gaon', 'Madanpura', 'Manjarli',
       'Sector35D Kharghar', 'Mumbai', 'Veera Desai Road',
       'Mira Bhayander Road', 'Mira Bhayandar', 'Lokhandwala Complex',
       'Old Panvel', 'Ekta Nagar', 'Adharwadi', 'Sector 6 Kamothe',
       'Kasar Vadavali', 'Yari Road', 'Lodhivali', 'Dattapada',
       'Natakwala Lane', 'Khanda Colony', 'Vidya Vihar',
       'Residential Flat Virar', 'Kasheli',
       'Oshiwara Police Station Road', 'Joveli Gaon', 'Charni Road',
       'Nagpada', 'Yogi Nagar', 'Lic Colony', 'Desale Pada', 'Bkc Bandra',
       'Sai Baba Nagar Lane', 'Gorai 1', 'Borivali Old Mhb Colony',
       'Chinchpada', 'Bolinj Naka', 'Sector 21 Kharghar', 'Babhai',
       'Pisavli Village', 'Shri Hari Nagar Kalyan', 'Vallabh Baug Lane',
       'Sector13 Kharghar', 'Century Mills', 'New Panvel Karanjade',
       'Juhu Tara', 'Nehru Nagar', 'Mulund Mumbai', 'Thane Diva',
       'Ram Mandir Road', 'Chinchpada Gaon', 'Nandivali Gaon',
       'Juhu Scheme', 'Vakola Santacuz E', 'Versova', 'Worli Sea Fase',
       'Rabale Station Road', 'Padle Gaon', 'Taloja Phase 2',
       'Sector 44 Seawoods', 'Kalina', 'Acc Cement Road',
       'Sector 2 Charkop Kandivali', 'Bolinj', 'Crawford Market',
       'Sion Koliwada', 'Dockyard Road', 'Shirgaon', 'Bhakti Park Wadala',
       'Ramchandra Lane', 'Charkop Sector 8', 'Siddhartha Nagar',
       'Beverly Park', 'Owale', 'Roadpali Navimumbai',
       'Sector 18 Kharghar', 'Vinobha Bhave Nagar Kurla', 'Satpati',
       'Vakola Pipeline Road', 'Umerkhadi', 'Santosh Nagar', 'Gorai 2',
       'Breach Candy', 'Maneklal Estate', 'Navapada',
       'Vakola Yashwant Nagar', 'Vijay Nagar', 'P L Lokhande Marg',
       'Vikhroli Park Site', 'Jari Mari', 'Tembhode Palghar', 'J B Nagar',
       'Vazira Naka', 'Mandvi', 'Bhuleshwar', 'Palava',
       'Sector 30 Kharghar', 'Sector 21 Nerul', 'Kon', 'Atgaon', 'Padgha',
       'Dahanu', 'Khar Danda', 'Vajreshwari', 'Kolegaon', 'Haware City',
       'Kolhare', 'Railwaycolony', 'Usarghar Gaon', 'Vakas', 'Uran',
       'Jvlr', 'Marol', 'Anand Nagar', 'Chandivali', 'Thakur Village',
       'Siddharth Nagar, Goregaon', 'Eksar', 'Hiranandani Gardens Powai',
       'Oshiwara', 'Battipada', 'Nehru Nagar, Kanjurmarg',
       'Lokhandwala Andheri', 'Dindoshi', 'Charkop', 'Jvpd Scheme',
       'Magathane', 'New Rajaram Wadi', 'Adarsh Nagar', 'Subhash Nagar',
       'Chakala', 'Lalbaug', 'Sector 8 Charkop', 'Dn Nagar', 'Hariyali',
       'Sindhi Society Chembur', 'Kannamwar Nagar 1', 'Sakinaka',
       'Evershine Nagar', 'Sambhaji Nagar', 'Anand Nagar,Dahisar',
       'Chincholi Bunder', 'Ghati Pada', 'Tagore Nagar',
       'Gokuldham Colony', 'Kanjur Village', 'Lbs Marg', 'Mogra Pada',
       'Sher E Punjab Colony', 'Shreyas Colony', 'Pant Nagar']

    property_type=['Apartment','Villa','Independent House','Independent Floor','Studio Apartment']

    furnished=['Unfurnished','Furnished','Semi-Furnished']

    locality=st.selectbox("Locality",locality)
    property_type=st.selectbox("Property Type",property_type)
    furnished=st.selectbox("Furnished",furnished)

    area=st.number_input("Total area in sqft",0.0,25000.0,100.0)
    bedroom_num=st.slider("BHK",1,20,1)
    bathroom_num=st.slider("Bathroom number",1,20,1)
    balcony_num=st.slider("Balcony number",0,20,0)
    age=st.slider("Age of property",0,100,2)
    total_floors=st.slider("Total Floors",1,100,1)
    
    ok=st.button("Calculate Price")
    if ok:
        X=pd.DataFrame({
            "area": [area],
            "locality":[locality],
            "property_type":[property_type],
            "bedroom_num":[bedroom_num],
            "bathroom_num":[bathroom_num],
            "balcony_num":[balcony_num],
            "furnished":[furnished],
            "age":[age],
            "total_floors":[total_floors]

        })
        
        price_per_sqft=data.predict(X)
        total_price=price_per_sqft*area
        st.subheader(f"The estimated price per sqft is {price_per_sqft[0]:.2f}")
        st.subheader(f"The estimated price is {total_price[0]:.2f}")
        
        


