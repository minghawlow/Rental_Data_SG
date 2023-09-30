import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

# Set the Streamlit app width to the maximum available width
st.set_page_config(
        page_title="Singapore Rental Property Dashboard",
        page_icon="house_buildings",
        layout="wide",
    )

header = st.container()
chart = st.container()
map = st.container()
modeltraining = st.container()

with header:
    st.title("Singapore Rental Property Dashboard")

    # Load data and cache it
    @st.cache_data
    def load_data():
        return pd.read_csv('Rental_Project/Streamlit_Dashboard/Rental_data_final.csv')

    df = load_data()

    # Create a metrics section to display the total number of listings
    total_listings = len(df)
    st.metric("Total Listings", total_listings)

with chart:
    # Split the app into two columns for the first row
    col1a, col1b = st.columns(2)
    with col1a:
        st.header("Average Rental Price by Planning Area")
        # Sidebar filters for the bar charts
        # st.sidebar.title("Filters - Bar Charts")
        selected_regions_bar = st.selectbox("Select Region(s)", ["All"] + df["region"].unique().tolist())

        # Filter data based on selected regions
        if "All" in selected_regions_bar:
            filtered_df_bar = df
        else:
            filtered_df_bar = df[df["region"]==selected_regions_bar]

        # Calculate the average rental price per planning area for the first bar chart
        average_prices_bar = filtered_df_bar.groupby("planning_area")["price"].mean().reset_index()
        average_prices_bar["price"] = average_prices_bar["price"].round()

        # Sort the data based on the user's choice for the first bar chart
        average_prices_bar = average_prices_bar.sort_values(by="price", ascending=False)

    
        # Create the bar chart figure with Plotly
        fig = px.bar(
            average_prices_bar,
            x="planning_area",
            y="price",
            orientation="v"
        )

        # Remove gridlines
        fig.update_xaxes(title_text= 'Planning Area', showgrid=False)
        fig.update_yaxes(title_text= 'Average Price (SGD)', showgrid=False)

        # Customize the chart
        fig.update_traces(
            marker_color="skyblue",
            hovertemplate='<span style="font-size:16px;"><b>%{x}</b></span><br><b>Average Price:</b> SGD %{y}',
            hoverlabel=dict(font=dict(size=12))
        )

        # Display the Plotly chart using st.plotly_chart
        st.plotly_chart(fig, use_container_width=True)

    with col1b:
        st.header("Distribution of Rental Price")
        st.markdown('#')
        st.markdown('###')
        st.markdown('###')


        # Filter data based on selected price range for the histogram
        filtered_data_hist = filtered_df_bar[(filtered_df_bar['price'] >= 0) & (filtered_df_bar['price'] <= 5000)]

        # Create the histogram with Plotly for the second chart
        fig = px.histogram(filtered_data_hist, x='price', nbins=30, range_x=(0,5000))
        fig.update_traces(marker=dict(color='skyblue', line=dict(color='white', width=1)))
        custom_tooltip_template = "<b>Price:</b> %{x} SGD<br><b>Number of Listings:</b> %{y}"
        fig.update_traces(hovertemplate=custom_tooltip_template,
                        hoverlabel=dict(font=dict(size=14)))
        fig.update_xaxes(title_text="Price (SGD)", showgrid=False)
        fig.update_yaxes(title_text="Number of Listings", showgrid=False)
        st.plotly_chart(fig)

with map:
    st.header("Map Distribution")
    col2a, col2b = st.columns([0.6,0.4])
    with col2b:
        # Filters for the map
        selected_area_map = st.selectbox("Select Area", ["All"] + list(df["planning_area"].unique()))
        unit_type_map = st.multiselect("Select Unit Type", df["unit_type"].unique(), default=df["unit_type"].unique())
        room_type_map = st.multiselect("Select Room Type", df["room_type"].unique(), default=df["room_type"].unique())
        price_range_map = st.slider("Select Price Range (SGD)", 0, 50000, (0, 5000))
        distance_map = st.slider("Select Distance to MRT (km)", 0.0, 5.0, (0.0, 1.0))

    with col2a:
        st.markdown("Use the filters to customize the map.")
        # Create the map for the second row
        # Filter the data based on user selections for the map
        filtered_df_map = df[
            (df["unit_type"].isin(unit_type_map))
            & (df["room_type"].isin(room_type_map))
            & (df["price"] >= price_range_map[0])
            & (df["price"] <= price_range_map[1])
            & (df["Distance_to_Nearest_MRT_km"] >= distance_map[0])
            & (df["Distance_to_Nearest_MRT_km"] <= distance_map[1])
        ]
        def calculate_bounding_box(selected_area):
            # Filter the DataFrame to get data points for the selected_area for the map
            if "All" in selected_area:
                area_data_map = filtered_df_map  
            else:
                area_data_map = filtered_df_map[filtered_df_map["planning_area"] == selected_area]

            # Calculate the bounding box
            min_lat = area_data_map["latitude"].min()
            max_lat = area_data_map["latitude"].max()
            min_lon = area_data_map["longitude"].min()
            max_lon = area_data_map["longitude"].max()

            # Calculate the center coordinates
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2

            # Calculate the zoom level
            if 'All' in selected_area:
                zoom_level_map = 10.2
            else:
                zoom_level_map = 13

            return {"center_lat": center_lat, "center_lon": center_lon, "zoom_level_map": zoom_level_map}

        # Calculate the bounding box and center based on the selected area for the map
        bounding_box_map = calculate_bounding_box(selected_area_map)

        # Create an interactive map using Plotly for the second row
        fig_map = px.scatter_mapbox(
            filtered_df_map,
            lat="latitude",
            lon="longitude",
            color="price",
            color_continuous_scale=px.colors.sequential.Blues,
            title="",
            mapbox_style="carto-positron",
            zoom=bounding_box_map["zoom_level_map"],
            center={"lat": bounding_box_map["center_lat"], "lon": bounding_box_map["center_lon"]}
        )
        marker_size_map = [10 if bounding_box_map["zoom_level_map"] < 12 else 20] * len(filtered_df_map)
        fig_map.update_traces(marker=dict(size=marker_size_map))
        fig_map.update_layout(
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Price (SGD)"),
        )
        fig_map.update_traces(
            hovertemplate='<span style="font-size:18px;"><b>%{customdata[0]}</b></span><br><b>Region:</b> %{customdata[1]}<br><b>Planning Area:</b> %{customdata[2]}<br><b>Unit Type:</b> %{customdata[3]}<br><b>Room Type:</b> %{customdata[4]}<br><b>Price:</b> %{customdata[5]} SGD<br><a href="%{customdata[6]}" target="_blank">Link to Address</a>',
            customdata=filtered_df_map[["address", "region","planning_area","unit_type", "room_type", "price","link"]],
                hoverlabel=dict(font=dict(size=14))
            )

        # Display the interactive map
        st.plotly_chart(fig_map, use_container_width=True)

with modeltraining:
    st.header("Rental Price Prediction")
    
    #Data processing
    @st.cache_data
    def preprocess_data(data):
        data = data.drop(columns=["address", "full_address", "updated_date", "link", "status"])

        true_numerical_cols = ["latitude", "longitude", "latitude_mrt", "longitude_mrt", 
                            "Distance_to_Nearest_MRT_km", "Walking_Time_to_Nearest_MRT_min", 
                            "room_size_sqft"]
        
        # Initialize the scaler
        scaler = StandardScaler()
        
        # Fit and transform the scaler to normalize the numerical features
        data[true_numerical_cols] = scaler.fit_transform(data[true_numerical_cols])
        
        one_hot_cols = ["region", "unit_type", "room_type"]
        ordinal_cols = ["planning_area", "Nearest_MRT_Station"]
        
        # Initialize encoders
        one_hot_encoder = OneHotEncoder(drop="first", sparse=False)
        ordinal_encoder = OrdinalEncoder()
        
        # Apply ordinal encoding to the specified columns
        data[ordinal_cols] = ordinal_encoder.fit_transform(data[ordinal_cols])
        
        # Apply one-hot encoding to the specified columns
        one_hot_encoded = one_hot_encoder.fit_transform(data[one_hot_cols])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols))
        
        # Concatenate the one-hot encoded dataframe with the original dataframe
        data = pd.concat([data, one_hot_encoded_df], axis=1)
        
        # Drop the original categorical columns
        data.drop(one_hot_cols, axis=1, inplace=True)

        return data, scaler, ordinal_encoder, one_hot_encoder

    data, scaler, ordinal_encoder, one_hot_encoder = preprocess_data(df)

    # Splitting dataset
    @st.cache_data
    def split_data(data):
        X = data.drop("price", axis=1)
        y = data["price"]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train, X_test, y_train, y_test = split_data(data)

    @st.cache_data
    def train_random_forest(X_train, y_train):
        rf = RandomForestRegressor(
            max_depth=36,
            max_samples=0.8898762217860171,
            max_features='log2',
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=1401,
            random_state=42
        )
        rf.fit(X_train, y_train)
        return rf
    
    rf = train_random_forest(X_train, y_train)
    
    # Load the trained machine learning model
    # model = joblib.load("/Users/jameslow/Downloads/rf_model.pkl")

    # Load the scaler for numerical features
    # scaler = joblib.load("/Users/jameslow/Downloads/scaler.pkl")

    # Load the saved OrdinalEncoder and OneHotEncoder
    # ordinal_encoder = joblib.load("/Users/jameslow/Downloads/ordinal_encoder.pkl")
    # one_hot_encoder = joblib.load("/Users/jameslow/Downloads/one_hot_encoder.pkl")

    cola, colb, colc, cold = st.columns(4)
    with cola:
    # Select Region
        selected_region = st.selectbox("Select Region", df["region"].unique())

    with colb:
    # Filter planning areas based on the selected region
        planning_areas_in_selected_region = df[df["region"] == selected_region]["planning_area"].unique()
    #Select Planning Area
        selected_planning_area = st.selectbox("Select Planning Area", planning_areas_in_selected_region)

    with colc:
    # Select Unit Type
        selected_unit_type = st.selectbox("Select Unit Type", df["unit_type"].unique())

    with cold:
    # Select Room Type
        selected_room_type = st.selectbox("Select Room Type", df["room_type"].unique())

    # Assuming you have already selected region, planning_area, unit_type, and room_type
    selected_properties = df[(df["region"] == selected_region) & (df["planning_area"] == selected_planning_area)]

    # Calculate statistics for remaining features based on selected properties
    mean_latitude = selected_properties["latitude"].mean()
    mean_longitude = selected_properties["longitude"].mean()
    mean_latitude_mrt = selected_properties["latitude_mrt"].mean()
    mean_longitude_mrt = selected_properties["longitude_mrt"].mean()

    # Handle missing values by using default values or other strategies
    mean_room_size_sqft = selected_properties[selected_properties["room_type"] == selected_room_type]["room_size_sqft"].median()
    mode_nearest_mrt_station = selected_properties["Nearest_MRT_Station"].mode().iloc[0]
    mean_distance_to_nearest_mrt = selected_properties["Distance_to_Nearest_MRT_km"].mean()
    mean_walking_time_to_nearest_mrt = selected_properties["Walking_Time_to_Nearest_MRT_min"].mean()

    # Create input data with default values
    input_data = pd.DataFrame({
        "region": [selected_region],
        "planning_area": [selected_planning_area],
        "unit_type": [selected_unit_type],
        "room_type": [selected_room_type],
        "latitude": [mean_latitude],
        "longitude": [mean_longitude],
        "latitude_mrt": [mean_latitude_mrt],
        "longitude_mrt": [mean_longitude_mrt],
        "room_size_sqft": [mean_room_size_sqft],
        "Nearest_MRT_Station": [mode_nearest_mrt_station],
        "Distance_to_Nearest_MRT_km": [mean_distance_to_nearest_mrt],
        "Walking_Time_to_Nearest_MRT_min": [mean_walking_time_to_nearest_mrt]
        # Add other features as needed
    })

    # Scale numerical features using the loaded scaler
    numerical_features = ["latitude", "longitude", "latitude_mrt", "longitude_mrt",
                        "Distance_to_Nearest_MRT_km", "Walking_Time_to_Nearest_MRT_min",
                        "room_size_sqft"]
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    one_hot_cols = ["region", "unit_type", "room_type"]
    ordinal_cols = ["planning_area", "Nearest_MRT_Station"]
        
    # Apply ordinal encoding to the specified columns
    input_data[ordinal_cols] = ordinal_encoder.transform(input_data[ordinal_cols])

    # Apply one-hot encoding to the specified columns
    one_hot_encoded = one_hot_encoder.transform(input_data[one_hot_cols])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols))

    # Concatenate the one-hot encoded dataframe with the original dataframe
    input_data = pd.concat([input_data, one_hot_encoded_df], axis=1)

    # Drop the original categorical columns
    input_data.drop(one_hot_cols, axis=1, inplace=True)

    if st.button("Predict"):
        try:
            # Make predictions using the loaded model
            prediction = rf.predict(input_data)
            st.success(f"Predicted Price: ${prediction[0]:.2f}")
        except ValueError as e:
            # Handle the ValueError (NaN or infinity) with a warning message
            st.warning("Sorry, there is no historical data for such unit.")
