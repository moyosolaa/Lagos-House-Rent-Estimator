import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


def load_data(dataset):
    df = pd.read_csv(dataset)
    return df


bedroom = {'1': 1, '2': 2, '3': 3, '4': 4,
           '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

bathrooms = {'1': 1, '2': 2, '3': 3, '4': 4,
             '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

toilets = {'1': 1, '2': 2, '3': 3, '4': 4,
           '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

location_rank = {'Ajah': 'ajah', 'Gbagada': 'gbagada', 'Ikeja': 'ikeja', 'Ikorodu': 'ikorodu',
                 'Ikoyi': 'ikoyi', 'Iyana Ipaja': 'iyanaipaja', 'Lekki': 'lekki', 'Ogba': 'ogba', 'Surulere': 'surulere', 'Yaba': 'yaba'}

estate = {'No': '0', 'Yes': '1'}

terraced = {'No': '0', 'Yes': '1'}

New_flag = {'No': '0', 'Yes': '1'}

security_flag = {'No': '0', 'Yes': '1'}

exec_flag = {'Low Income Range': '1', 'Middle Range 1': '2',
             'Middle Range 2': '3', ' Expensive Highbrow Areas': '4'}

serviced_flag = {'No': '0', 'Yes': '1'}


# Get the keys in dictionary

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return str(value)

# Find the keys in the dictionary

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


modelz = joblib.load("67k_model.pkl")


def main():
    """Housing Ml App"""
    st.title("House Rent Pricing App")
#     st.markdown('<style>h1{color:blue;},/style>', unsafe_allow_html=True)
    st.subheader(
        "Built for major cities in Lagos State, other locations will be supported soon")
    st.markdown('<style>h1{color:red;},/style>', unsafe_allow_html=True)

#     menu = ["Prediction"]
#     choices = st.sidebar.selectbox("Select Activities",menu)
    st.info("With access to over 60,000 listings from www.propertypro.ng, we're able to provide you with estimated prices for the very kind of house you want so you can update your budget accurately. We also provide you with a list of properties that fit right into these your budget even before you start hunting. All just to make your life easier.")


#     if choices == 'Prediction':
    st.sidebar.info("Select features you want in your house below: ")
    st.subheader("Prediction")
    st.markdown("To make a prediction, select house features on the sidebar and click on Evaluate below.")
    locations = st.sidebar.selectbox(
        "Your preferred location to live", tuple(location_rank.keys()))
    exec_flagg1 = st.sidebar.selectbox(
        "What type of area do you want to live?", tuple(exec_flag.keys()))
    Number_of_Bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 9)
    Number_of_Bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 9)
    Number_of_Toilet = st.sidebar.slider("Number of Toilets", 1, 9)
    estate_or_not = st.sidebar.selectbox(
        "Do you want to live in an Estate?", tuple(estate.keys()))
    terrace_or_not = st.sidebar.selectbox(
        "Do you want to live in a Terraced Apartment?", tuple(terraced.keys()))
    Number_of_flag = st.sidebar.selectbox(
        "Do you prefer a new apartment?", tuple(New_flag.keys()))
    security_flag = st.sidebar.selectbox(
        "Do you prefer an apartment with security guards?", tuple(New_flag.keys()))
    serviced_flag1 = st.sidebar.selectbox(
        "Do you prefer an serviced apartment?", tuple(serviced_flag.keys()))

    # Encoding
    v_estate_or_not = get_value(estate_or_not, estate)
    v_locations = get_value(locations, location_rank)
    v_terrace_or_not = get_value(terrace_or_not, terraced)
    v_Number_of_flag = get_value(Number_of_flag, New_flag)
    v_security_flag = get_value(New_flag, New_flag)
    v_exec_flagg1 = get_value(exec_flagg1, exec_flag)
    v_serviced_flag1 = get_value(serviced_flag1, serviced_flag)

    vv_estate_or_not = get_key(estate_or_not, estate)
    vv_terrace_or_not = get_key(terrace_or_not, terraced)
    vv_Number_of_flag = get_key(Number_of_flag, New_flag)
    vv_security_flag = get_key(New_flag, New_flag)
    vv_exec_flagg1 = get_key(exec_flagg1, exec_flag)
    vv_serviced_flag1 = get_key(serviced_flag1, serviced_flag)

    # Function to convert Cleaned data csv and Location rank csv to Panda dataframe
    data = load_data('clean_data.csv')
    data2 = load_data('locationrank.csv')

    # Block of code that execute when you click evaluate
    if st.button("Evaluate"):
        # Join Location and Number of bedrooms to produce locationbed variable
        locationbed = v_locations + str(Number_of_Bedrooms)
        df1 = data2[(data2['locationbed'] == locationbed)]
        locationBedRank = df1.location_rank.values[0]

        # Arranging predictor data in the same way model was trained
        predictor_data = [Number_of_Bedrooms, Number_of_Bathrooms, Number_of_Toilet, v_estate_or_not, locationBedRank,
                          v_terrace_or_not, v_Number_of_flag, v_exec_flagg1, v_serviced_flag1]
        predictor_data = np.array(predictor_data).reshape(1, -1)
        z = data.where(data['location'] == v_locations)
        b = z.new_price.max()
        c = z.new_price.min()
        d = int(c)
        e = int(b)

        predicted = modelz.predict(predictor_data)
        predicted = int(predicted)
        st.write("The predicted price for an apartment in ", locations, "with ", Number_of_Bedrooms, " bedrooms,",
                 Number_of_Bathrooms, " bathrooms and ", Number_of_Toilet, " toilets is estimated at ", predicted, " naira.")
        st.write("Most Expensive house in  ", locations, "costs about",
                 e, " naira.", ", while the least expensive house costs about", d, " naira.")


if __name__ == "__main__":
    main()