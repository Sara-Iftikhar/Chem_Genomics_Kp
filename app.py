
import cv2
import pandas as pd
import streamlit as st

from scripts.utils import extract_colony, crop_img


# Define grid cell dimensions (for example, based on manual measurement)
cell_width = 100  # Adjust this based on the actual image
cell_height = 100  # Adjust this based on the actual image

# Load strain data from an Excel file
strain_file_path = f'/Users/iftis0a/IDE/collaborations/Ge/Chem_Genomics_Kp/starain_names.xls'
strains_df = pd.read_excel(strain_file_path)

# Streamlit app layout
st.title("Colony Picker")

# Input row and column
# Choose method for colony extraction
method = st.sidebar.radio("Select colony extraction method:", ("By Strain Name", "By Row and Column"))

# Inputs based on selected method
if method == "By Strain Name":
    # Strain selection
    strain_name = st.sidebar.selectbox("Select Strain", strains_df['ID'])
    # Get row and column numbers for the selected strain
    strain_row_col = strains_df.loc[strains_df['ID'] == strain_name, ['Row', 'Column']].values[0]
    row = int(strain_row_col[0])
    col = int(strain_row_col[1])
else:
    # Manual row and column input
    row = st.sidebar.number_input("Enter Row:", min_value=0, value=0, step=1)
    col = st.sidebar.number_input("Enter Column:", min_value=0, value=0, step=1)

# Condition selection
condition = st.sidebar.selectbox("Pick one condition", ["Colistin-0.8ugml", "ethanol-5%", "Gentamycin-0,8ugml",
                                                        "Gentamycin-4ugml", "Gentamycin-8ugml",
                                                        "Imipenem-0.1ugml", "LB-", "LBBlood-5%", "LBUrine-5%",
                                                        "pH-8", "pH-9"])
strain_nums = ["A", "B", "C", "D", "E"]  # Condition numbers to display in parallel

# Button to extract and display colonies
if st.button("Extract Colonies"):
    if method == "By Strain Name":
        st.write(f"Displaying colonies for strain: **{strain_name}**")
    else:
        st.write(f"Displaying colonies for Row: **{row}**, Column: **{col}**")

    # Create a column layout for each condition number
    cols = st.columns(len(strain_nums))

    for i, strain_num in enumerate(strain_nums):
        # Load image path for each condition_num
        image_path = f'/Users/iftis0a/IDE/collaborations/Ge/Chem_Genomics_Kp/figures/{condition}-1-1_{strain_num}.JPG.grid.jpg'

        # Read image using OpenCV
        img = crop_img(image_path)

        # Extract the colony based on the input row and column
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

            # Display the extracted colony image for the current condition number in its column
            cols[i].image(colony_img_rgb, caption=f"Replicate {strain_num}", use_column_width=True)

# Instructions for the user
st.write(
    "Select a strain or enter row and column numbers, then click 'Extract Colonies' to view the colonies across all conditions.")
