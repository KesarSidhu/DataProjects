import streamlit as st

def meet_the_team():
    team_members = [
        {
            "name": "Sri Lakshmi Panda",
            "role": "Executive Tech Lead",
            "image": "Images/sri.jpeg"
        },
        {
            "name": "Ruba Thekkath",
            "role": "Tech Lead",
            "image": "Images/sri.jpeg"
        },
        {
            "name": "Manushri Rane",
            "role": "Tech Lead",
            "image": "Images/sri.jpeg"
        },
        {
            "name": "Kesar Siddu",
            "role": "Tech Lead",
            "image": "Images/sri.jpeg"
        },
        {
            "name": "Arya Trivedi",
            "role": "Tech Lead",
            "image": "Images/sri.jpeg"
        }
    ]

    st.markdown(
        """
        <style>
        .profile-icon {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            object-fit: cover;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Meet the Team")
    st.subheader("Get to know the talented individuals behind our project!")
    st.write("\n\n\n")

    cols = st.columns(3)  # Adjust the number of columns as needed

    for idx, member in enumerate(team_members):
        with cols[idx % 3]:  # Distribute team members across columns
            image_url = member["name"]
            st.markdown('<img src="{image_url}" class="profile-icon" alt="Profile Icon">', unsafe_allow_html=True)


            # st.image(member["image"], width=150)
            st.subheader(member["name"])
            st.write(member["role"])

    # Footer
    st.write("___")
    st.write("© 2024 GDSC Link. All rights reserved.")