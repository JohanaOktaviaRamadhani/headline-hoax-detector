SHARED_CSS = """
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #FAFAFA; }
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.section-title {
    font-size: 1.0rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 6px;
    padding-bottom: 6px;
    border-bottom: 2px solid #E5E7EB;
}
.badge { display:inline-block; padding:2px 10px; border-radius:999px;
         font-size:0.78rem; font-weight:600; margin:2px; }
.badge-blue  { background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; }
.badge-green { background:#F0FDF4; color:#15803D; border:1px solid #BBF7D0; }
.badge-red   { background:#FEF2F2; color:#B91C1C; border:1px solid #FECACA; }
.badge-gray  { background:#F3F4F6; color:#374151; border:1px solid #D1D5DB; }
.info-box {
    background: #F8FAFC;
    border-left: 4px solid #6366F1;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #374151;
}
.step-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.step-num {
    background: #6366F1;
    color: white;
    border-radius: 50%;
    width: 26px; height: 26px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 700;
    flex-shrink: 0;
}
.dict-row {
    display: flex; gap: 12px; padding: 8px 0;
    border-bottom: 1px solid #F3F4F6; align-items: flex-start;
}
.dict-col  { font-family:monospace; font-weight:600; color:#4F46E5;
             min-width:150px; font-size:0.88rem; }
.dict-type { color:#9CA3AF; font-size:0.78rem; min-width:60px; }
.dict-desc { color:#374151; font-size:0.88rem; }
.dataset-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.dataset-card-title {
    font-size: 1.0rem;
    font-weight: 700;
    color: #1F2937;
    margin-bottom: 6px;
}
</style>
"""

COLOR_MAIN       = "#6366F1"
COLOR_MENDUKUNG  = "#10B981"
COLOR_MEMBANTAH  = "#F43F5E"
COLOR_NEUTRAL    = "#94A3B8"
