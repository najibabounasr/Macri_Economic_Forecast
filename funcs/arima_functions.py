from statsmodels.tsa.stattools import adfuller
import pandas as pd
# Load Ethereum smart contract ABI and address (replace with your own values)
CONTRACT_ABI = 'your_smart_contract_abi'
CONTRACT_ADDRESS = 'your_smart_contract_address'

# Utility function to generate unique identifier for products
def generate_unique_identifier(product_name, product_desc):
    """
    Generate a unique identifier for a product using the product name, product description, and a random UUID.

    Args:
        product_name (str): The name of the product.
        product_desc (str): The description of the product.

    Returns:
        str: The unique identifier for the product.
    """

    # Generate a random UUID
    random_string = str(uuid.uuid4())

    # Concatenate the product name, product description, and the random UUID to create a unique identifier
    unique_identifier = f"{product_name}-{product_desc}-{random_string}"

    return unique_identifier

# Utility function to generate QR code
def generate_qr_code(input_data):
    """
    Generate a QR code image from the given input data.

    Args:
        input_data (str): The input data to be encoded in the QR code.

    Returns:
        qrcode.image.pil.PilImage: The generated QR code image.
    """

    # Create a QRCode object with desired parameters
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )



# # Generate a unique identifier for the product using its name and description
# unique_identifier = generate_unique_identifier(product_name, product_desc)

# # Create a QR code image containing the unique identifier
# qr_image = generate_qr_code(unique_identifier)

# # Display the generated QR code image in the Streamlit app, with a caption and using the column width
# st.image(qr_image, caption="Generated QR Code", use_column_width=True)







#     # Add the input data to the QRCode object
#     qr.add_data(input_data)

#     # Optimize the QR code data for the given input
#     qr.make(fit=True)

#     # Create an image from the QR code data
#     img = qr.make_image(fill_color="black", back_color="white")

#     return img


# Verify product authenticity using unique identifier
def verify_product(unique_identifier):
    """
    Verify a product's authenticity using its unique identifier.
    
    Args:
        unique_identifier (str): The unique identifier of the product.

    Returns:
        bool: True if the product is authentic, False otherwise.
        dict: The product information if the product is authentic, None otherwise.
    """
    product = contract.functions.getProduct(unique_identifier).call()
    if product:
        return True, product
    else:
        return False, None


# Define a function to perform the Augmented Dickey-Fuller test
def check_stationarity(data):
    """
    Perform Augmented Dickey-Fuller test to check for stationarity.
    
    Arguments:
    Pandas Series: a series of data to be checked for stationarity.
    
    Returns:
    Prints test statistics and critical values.
    """
    # Perform Augmented Dickey-Fuller test
    # Perform the test using the AIC criterion for choosing the number of lags
    print('Results of Augmented Dickey-Fuller Test:')
    adf_test = adfuller(data, autolag='AIC')  

    # Extract and print the test statistics and critical values
    adf_output = pd.Series(adf_test[0:4], 
                           index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    for key, value in adf_test[4].items():
        adf_output['Critical Value (%s)' % key] = value
    print(adf_output)
    return adf_output