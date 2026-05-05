import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import SHARED_CSS

st.set_page_config(page_title="Dataset Lain · Hoax Detector", page_icon="📰", layout="wide")
st.markdown(SHARED_CSS, unsafe_allow_html=True)

st.markdown("## 📰 Dataset Lain")
st.markdown(
    "<div class='info-box'>Halaman ini akan diisi ketika dataset berikutnya tersedia.</div>",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("""
    <div style='text-align:center;padding:60px 40px;background:#FFFFFF;
                border:2px dashed #E5E7EB;border-radius:16px'>
        <div style='font-size:3rem;margin-bottom:16px'>🔜</div>
        <div style='font-size:1.2rem;font-weight:700;color:#1F2937;margin-bottom:8px'>
            Coming Soon
        </div>
        <div style='color:#6B7280;font-size:0.9rem;line-height:1.6'>
            Halaman ini sedang disiapkan.<br>
            Dataset berikutnya akan ditampilkan di sini setelah tersedia.
        </div>
    </div>
    """, unsafe_allow_html=True)
