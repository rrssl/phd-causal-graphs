import numpy as np
from scipy.stats import linregress


x = np.arange(1, 10)
y = np.array([3, 7, 10, 13, 17, 20, 24, 26, 29])
print(linregress(x, y))
# slope=3.2666666666666666, intercept=0.22222222222222499, rvalue=0.99847634789264694, pvalue=4.5393523643395994e-10, stderr=0.068235508762554298
