import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the cleaned (non-encoded) dataset to extract the features and their unique values
hotel_df = pd.read_csv('/Users/kesarsidhu/VS Code/AISC/BP - Winter 2025/cleaned_hotel_data.csv')

# Load the trained XGBoost model
with open('rf_best_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# App title and intro
st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("This app predicts whether a hotel booking is likely to be **canceled** or **honored** based on user inputs.")

st.subheader("Enter Booking Details Below:")

# Numerical inputs with values and help text
lead_time = st.number_input(
    "Lead Time (days)", 
    min_value=0, 
    max_value=hotel_df['lead_time'].max(), 
    step=1,
    help="Number of days between booking and arrival"
)

arrival_date_month = st.number_input(
    "Arrival Date Month (1-12)", 
    min_value=1, 
    max_value=12, 
    step=1,
    help="Month of arrival"
)

arrival_date_week_number = st.number_input(
    "Arrival Week Number", 
    min_value=1, 
    max_value=53, 
    step=1,
    help="Week number of the year for arrival"
)

arrival_date_day_of_month = st.number_input(
    "Arrival Day of Month", 
    min_value=1, 
    max_value=31, 
    step=1,
    help="Day of the month of arrival"
)

stays_in_weekend_nights = st.number_input(
    "Stays in Weekend Nights", 
    min_value=0, 
    max_value=hotel_df['stays_in_weekend_nights'].max(), 
    step=1,
    help="Number of weekend nights (Saturday and Sunday) the guest stayed or booked"
)

stays_in_week_nights = st.number_input(
    "Stays in Week Nights", 
    min_value=0, 
    max_value=hotel_df['stays_in_week_nights'].max(), 
    step=1,
    help="Number of weeknights the guest stayed or booked"
)

adults = st.number_input(
    "Number of Adults", 
    min_value=1, 
    max_value=hotel_df['adults'].max(), 
    step=1,
    help="Number of adults"
)

children = st.number_input(
    "Number of Children", 
    min_value=0, 
    max_value=int(hotel_df['children'].max()), 
    step=1,
    help="Number of children"
)

babies = st.number_input(
    "Number of Babies", 
    min_value=0, 
    max_value=hotel_df['babies'].max(), 
    step=1,
    help="Number of babies"
)

is_repeated_guest = st.selectbox(
    "Is Repeated Guest?", 
    options=[0, 1],
    help="If the guest is a repeat customer (0 = Not Repeated, 1 = Repeated)"
)

previous_cancellations = st.number_input(
    "Previous Cancellations", 
    min_value=0, 
    max_value=hotel_df['previous_cancellations'].max(), 
    step=1,
    help="Number of previous bookings that were canceled by the customer"
)

previous_bookings_not_canceled = st.number_input(
    "Previous Bookings Not Canceled", 
    min_value=0, 
    max_value=hotel_df['previous_bookings_not_canceled'].max(), 
    step=1,
    help="Number of previous bookings that were not canceled by the customer"
)

booking_changes = st.number_input(
    "Booking Changes", 
    min_value=0, 
    max_value=hotel_df['booking_changes'].max(), 
    step=1,
    help="Number of changes made to the booking"
)

days_in_waiting_list = st.number_input(
    "Days in Waiting List", 
    min_value=0, 
    max_value=hotel_df['days_in_waiting_list'].max(), 
    step=1,
    help="Number of days the booking was in the waiting list"
)

adr = st.number_input(
    "Average Daily Rate (ADR)", 
    min_value=0.0, 
    max_value=hotel_df['adr'].max(), 
    step=1.0,
    help="Average Daily Rate (ADR) for the booking"
)

required_car_parking_spaces = st.number_input(
    "Required Car Parking Spaces", 
    min_value=0, 
    max_value=hotel_df['required_car_parking_spaces'].max(), 
    step=1,
    help="Number of car parking spaces required"
)

total_of_special_requests = st.number_input(
    "Total Special Requests", 
    min_value=0, 
    max_value=hotel_df['total_of_special_requests'].max(), 
    step=1,
    help="Number of special requests made by the guest"
)

number_of_bookings = st.number_input(
    "Number of Bookings", 
    min_value=1, 
    max_value=hotel_df['number_of_bookings'].max(), 
    step=1,
    help="Total number of bookings made by the guest"
)

# Categorical unique values extracted from the dataset to populate the selectbox options
deposit_type_options = hotel_df['deposit_type'].unique().tolist()
market_segment_options = hotel_df['market_segment'].unique().tolist()
customer_type_options = hotel_df['customer_type'].unique().tolist()
distribution_channel_options = hotel_df['distribution_channel'].unique().tolist()
meal_options = hotel_df['meal'].unique().tolist()
hotel_options = hotel_df['hotel'].unique().tolist()
reserved_room_type_options = hotel_df['reserved_room_type'].unique().tolist()

# Split the categorical features into two columns for a cleaner layout
col1, col2 = st.columns(2)  

with col1:
    deposit_type = st.selectbox(
        "Deposit Type", 
        options=deposit_type_options,
        help="Type of deposit made:\n- No Deposit: No deposit required\n- Refundable: Fully refundable deposit\n- Non Refund: Non-refundable deposit"
    )

    market_segment = st.selectbox(
        "Market Segment", 
        options=market_segment_options,
        help="Market segment designation"
    )

    customer_type = st.selectbox(
        "Customer Type", 
        options=customer_type_options,
        help="Type of customer:\n- Transient: Individual traveler, short-term stay\n- Contract: Contract-based stay\n- Transient-Party: Group of transient guests\n- Group: Large group booking"
    )

    reserved_room_type = st.selectbox(
        "Reserved Room Type", 
        options=reserved_room_type_options,
        help="Type of room reserved by the guest"
    )

with col2:
    distribution_channel = st.selectbox(
        "Distribution Channel", 
        options=distribution_channel_options,
        help="Booking distribution channel"
    )

    meal = st.selectbox(
        "Meal Type", 
        options=meal_options,
        help="Type of meal booked:\n- BB: Bed & Breakfast\n- FB: Full Board (includes breakfast, lunch, and dinner)\n- HB: Half Board (includes breakfast and one other meal)\n- SC: Self Catering (no meals included)\n- Undefined: No meal type specified"
    )

    hotel = st.selectbox(
        "Hotel Type", 
        options=hotel_options,
        help="Type of hotel (e.g., Resort Hotel, City Hotel)"
    )

# Take user inputs and encode them using the dictionary created earlier
encoded_features = {
    "deposit_type": deposit_type_options,
    "market_segment": market_segment_options,
    "customer_type": customer_type_options,
    "distribution_channel": distribution_channel_options,
    "meal": meal_options,
    "hotel": hotel_options,
    "reserved_room_type": reserved_room_type_options
}

encoded_inputs = {key: {val: idx for idx, val in enumerate(values)} for key, values in encoded_features.items()}

deposit_type = encoded_inputs["deposit_type"][deposit_type]
market_segment = encoded_inputs["market_segment"][market_segment]
customer_type = encoded_inputs["customer_type"][customer_type]
distribution_channel = encoded_inputs["distribution_channel"][distribution_channel]
meal = encoded_inputs["meal"][meal]
hotel = encoded_inputs["hotel"][hotel]
reserved_room_type = encoded_inputs["reserved_room_type"][reserved_room_type]

# Take all inputs and encoded values into a list for prediction (reshaped to 2D array since ML model expects 2D input)
input_data = np.array([
    hotel, lead_time, arrival_date_month, arrival_date_week_number, arrival_date_day_of_month,
    stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, meal, market_segment,
    distribution_channel, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled,
    reserved_room_type, booking_changes, deposit_type, days_in_waiting_list, customer_type, adr,
    required_car_parking_spaces, total_of_special_requests, number_of_bookings]).reshape(1, -1)

# Prediction section with a button identifying if the booking is likely to be canceled or not
if st.button("Predict Booking Cancellation"):
    prediction = rf_model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("❌ Prediction: Booking is likely to be **Canceled**.")
    else:
        st.success("✅ Prediction: Booking is likely to be **Not Canceled**.")
