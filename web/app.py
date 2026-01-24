import os
import sys
import streamlit as st

# Add project root to path (assuming run from root or web/)
# If run from root, os.path.dirname(__file__) is web/, parent is root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

try:
    from web.api_bridge import run_ndid
except ImportError:
    # If path resolving fails, try relative import if run from inside web/
    from api_bridge import run_ndid

st.set_page_config(
    page_title="Dejaview",
    layout="wide"
)

st.title("Dejaview")
st.subheader("Near Duplicate Image Detection (NDID)")
st.caption(
    "Upload an image to check whether a near-duplicate exists in the dataset."
)

st.divider()

left, right = st.columns([1, 1])

with left:
    st.markdown("### Input")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file:
        st.image(
            uploaded_file,
            caption="Uploaded Image",
            width=260
        )

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Run NDID", use_container_width=True)

with right:
    st.markdown("### Output")

    if uploaded_file and run_btn:
        with st.spinner("Analyzing image…"):
            result = run_ndid(uploaded_file)

        st.success("Analysis completed")

        st.markdown("#### Result Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Status", result["status"])
            st.metric("Method", result["method"])

        with col2:
            st.metric("Similarity", f"{result['similarity_percentage']:.2f}%")
        
        if result.get("message"):
            st.warning(result["message"])

        st.markdown("#### Matched Image")
        if result.get("matched_image_path"):
            st.image(
                result["matched_image_path"],
                caption="Most similar image found",
                width=300
            )
        else:
            st.info("No matching image found.")

    else:
        st.info("Upload an image and click **Run NDID** to see results.")
