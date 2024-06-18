import streamlit as st

def history():
    st.title("History of Development")

    st.markdown('<p class="large-font">This page provides an overview of the key milestones and phases in the development of our project.</p>', unsafe_allow_html=True)

    st.subheader("Process")
    st.write("Stages of development:")
    st.code("""
        - 1. Conceptualization
        - 2. Planning
        - 3. Design
        - 4. ML Model Development
        - 5. Training
        - 6. Testing
        - 7. Website / UI/UX / Website
    """, language='python')

    st.subheader("Challenges")
    st.write("We faced many challenges along the way and here are some of them:")
    st.code("""
        - Colab
            - Combining individual scripts into one
            - File-path errors due to Shared drive permission settings
        - Machine learning
            - Learning various preprocessing steps
            - Figuring out which pre-trained model to use
            - Increasing model's accuracy
        - Website
            - New technologies like streamlit
        - Other
            - Long running time
            - Midterms
    """, language='python')

    st.subheader("Learnings")
    st.write("Through this project we learnt about many new technologies and processes:")
    st.code("""
        - New technologies
            - Google Colab
            - Tensorflow
            - ResNet
            - Streamlit
        - Working with an actual team on a project and strategizing the splitting of work
        - Learning about python libraries
        - Learning how to overcome setbacks during coding
        - How to balance classes along with the club
    """, language='python')

    # Footer
    st.write("___")
    st.write("© 2024 GDSC Link. All rights reserved.")