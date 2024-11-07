
import numpy as np
def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)



vec = np.array([-4.79, -0.38, -0.001, 0.011, 0.4444, 2.34341232, 6.999,1.1])

trunc_vec=trunc(vec, decs=3)

print(trunc_vec)
