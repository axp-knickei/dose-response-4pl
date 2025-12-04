import pandas as pd
import numpy as np
import re

def process_csv(input_file, output_file):
    # 1. Load the CSV file
    # We use sep=';' because your file uses semicolons as delimiters
    df = pd.read_csv("2501203 - eal_meta-abs.csv", sep=';')

    # 2. Define the transformation function
    def convert_to_log(val):
        # If the value is missing (NaN), return it as is
        if pd.isna(val):
            return val
        
        # Ensure the value is treated as a string for regex matching
        val_str = str(val)
        
        # Regex to find the number before 'uM'
        # This looks for digits (including decimals) at the start of the string
        match = re.search(r'^(\d+(\.\d+)?)uM_', val_str)
        
        if match:
            try:
                # Extract the number part (e.g., '10' from '10uM')
                concentration = float(match.group(1))
                
                # Calculate the logarithm (log10)
                if concentration > 0:
                    return np.log10(concentration)
                else:
                    return val # Return original if concentration is 0 or invalid
            except ValueError:
                return val # Return original if conversion fails
        
        # If the pattern 'uM_5FU00' is not found (e.g., NegControl), return original
        return val

    # 3. Apply the function to columns '1' through '10'
    # We generate a list of column names: ['1', '2', ..., '10']
    cols_to_transform = [str(i) for i in range(1, 11)]

    for col in cols_to_transform:
        # Check if the column exists in the file before processing
        if col in df.columns:
            df[col] = df[col].apply(convert_to_log)

    # 4. Save the result to a new CSV file
    df.to_csv(output_file, sep=';', index=False)
    print(f"Successfully converted and saved to {output_file}")

# Run the function
# Ensure the input file name matches your file
process_csv('meta-abs4.csv', 'converted_meta-abs4.csv')