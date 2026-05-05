import streamlit as st

st.set_page_config(
    page_title="Hoax Detector Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #FAFAFA; }
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}
</style>
""", unsafe_allow_html=True)

st.switch_page("C:\\Users\\hanao\\Downloads\\hoax-detector\\ds-role\\app\\pages\\1_Overview.py")
