
import cv2
import streamlit as st

from scripts.utils import extract_colony, crop_img


# Define grid cell dimensions (for example, based on manual measurement)
cell_width = 100  # Adjust this based on the actual image
cell_height = 100  # Adjust this based on the actual image

# Streamlit app layout
st.title("Colony Picker")

# Input row and column
row = st.sidebar.number_input("Enter Row:", min_value=0, value=0, step=1)
col = st.sidebar.number_input("Enter Column:", min_value=0, value=0, step=1)
condition = st.sidebar.selectbox("Pick one", ["Colistin-0.8ugml", "ethanol-5%", "Gentamycin-0,8ugml",
                                              "Gentamycin-4ugml", "Gentamycin-8ugml",
                                              "Imipenem-0.1ugml", "LB-", "LBBlood-5%", "LBUrine-5%",
                                              "pH-8", "pH-9"])

condition_num = st.sidebar.selectbox("Pick one", ["A", "B", "C", "D", "E"])

# Load image
image_path = f'/Users/iftis0a/IDE/collaborations/kp_app/figures/{condition}-1-1_{condition_num}.JPG.grid.jpg'

# Read image using OpenCV
img = crop_img(image_path)

# Extract the colony based on the input row and column
if st.button("Extract Colony"):
    # Extract the colony at the specified row and column
    colony_img = extract_colony(img, row, col, cell_width, cell_height)

    if colony_img is not None:
        # Convert the image from BGR to RGB
        colony_img_rgb = cv2.cvtColor(colony_img, cv2.COLOR_BGR2RGB)

        # Convert image to grayscale
        gray = cv2.cvtColor(colony_img_rgb, cv2.COLOR_BGR2GRAY)

        # Create a binary mask where the black areas (background) are white, and everything else is black
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Invert the mask to get the background
        mask_inv = cv2.bitwise_not(mask)

        # Convert the original image to BGRA (with alpha channel)
        image_bgra = cv2.cvtColor(colony_img_rgb, cv2.COLOR_BGR2BGRA)

        # Set the alpha channel of the pixels in the background (mask_inv) to 0 (transparent)
        image_bgra[:, :, 3] = mask

        # Display the extracted colony image
        st.image(colony_img_rgb, caption=f"Colony at Row {row}, Column {col}", use_column_width=True)

# Instructions for the user
st.write("Enter the row and column numbers, then click 'Extract Colony' to view the corresponding colony.")
