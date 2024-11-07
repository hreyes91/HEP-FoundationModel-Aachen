
import pandas as pd
import os
import numpy as np

import numpy as np



def GenerateCodebook():
    # Define the ranges
    range1 = np.arange(40, -1, -1)  # [40, 0]
    range2 = np.arange(30, -1, -1)  # [30, 0]
    range3 = np.arange(30, -1, -1)

    # Use meshgrid to get all combinations of the ranges
    x, y, z = np.meshgrid(range1, range2, range3, indexing='ij')

    # Stack the results along a new axis and reshape
    combinations = np.stack([x, y, z], axis=-1)

    # Reshape to the required format (n_combinations, 1, 3)
    result = combinations.reshape(-1, 1, 3)


    print(result)
    # Display the shape of the result
    print(result.shape)

    return result
    

def TransformToPandas(result):

    reshaped_combinations = result.reshape(-1, 3)

    # Create a pandas DataFrame with the reshaped combinations
    df = pd.DataFrame(reshaped_combinations, columns=["PT_0", "Eta_0", "Phi_0"])
    print(df.head())
    
    return df





def SaveCodebook():

    df.to_hdf(os.path.join(".", f"samples_codebook_1const.h5"), key="discretized")

    return

result=GenerateCodebook()
df=TransformToPandas(result)
SaveCodebook()
