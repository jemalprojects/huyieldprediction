import streamlit as st

from  PIL import Image
icon = Image.open("icons.png")
st.set_page_config(
    page_title="Crop Yield Prediction",
    # page_icon="🌾",
    page_icon=icon,
    # layout="wide"
)
custom_css="""
<style>
header{
display:none !important;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.css-d1b1ld.edgvbvh6{
visibility:hidden;
}
.css-1v8iw71.eknhn3m4{
visibility:hidden;
}
.st-emotion-cache-6qob1r{
display:none !important;
}
.stSidebar{
display:none !important;
}
.st-emotion-cache-19ee8pt{
display:none !important;
}
._profileContainer_gzau3_53{
display:none !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)



st.subheader("ML Based Crop Yield Prediction", divider=True)
st.markdown("""
<p>This interface allows users to input key agricultural data, such as:</p>
<ul>
    <li><strong>District:</strong> Select the district where the crop is being cultivated.</li>
    <li><strong>Season:</strong> Choose the season of cultivation (e.g., rainy season, dry season).</li>
    <li><strong>Area (in square meters):</strong> Enter the area under cultivation.</li>
</ul>
<p>The model will automatically estimate the following parameters:</p>
<ul>
    <li><strong>GWETPROF, GWETTOP, GWETROOT:</strong> Soil moisture-related parameters.</li>
    <li><strong>CLOUD_AMT:</strong> Cloud cover amount.</li>
    <li><strong>TS:</strong> Soil temperature.</li>
    <li><strong>PS:</strong> Air pressure.</li>
    <li><strong>RH2M:</strong> Relative humidity at 2 meters.</li>
    <li><strong>QV2M:</strong> Specific humidity at 2 meters.</li>
    <li><strong>PRECTOTCORR:</strong> Corrected total precipitation.</li>
    <li><strong>T2M_MAX, T2M_MIN:</strong> Maximum and minimum 2-meter air temperature.</li>
    <li><strong>T2M_RANGE:</strong> Temperature range at 2 meters.</li>
    <li><strong>WS2M:</strong> Wind speed at 2 meters.</li>
</ul>
<p>After entering the district, season, and area, click the <span class="interactive-text">"Predict Crop Yield"</span> button to get the estimated crop yield.</p>""", unsafe_allow_html=True)


custom_css1="""
<style>
footer {
	
	visibility: hidden;
	
	}
footer:after {
	content:'goodbye'; 
	visibility: visible;
	display: block;
	position: relative;
	#background-color: red;
	padding: 5px;
	top: 2px;
}
</style>
"""
st.markdown(custom_css1, unsafe_allow_html=True)
