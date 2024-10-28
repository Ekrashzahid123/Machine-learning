# we are going to work with the linear regression in pyhton
import numpy as np
from sklearn.linear_model import LinearRegression
# now you have all the functionality to implement the linear regression and polynomial regression
#step 3 is to give data to the model
x =np.array([5,15,25,35,45,55]).reshape(-1,1)
y = np.array([5, 20, 14, 32, 22, 38])
#defining the model 
model=LinearRegression()
#calculating the optimum value for the data 
model.fit(x,y)                            #model=linearRegression.fit(x,y)
r_sq=model.score(x,y)
print(f"coefficient of determination: {r_sq}") # type: ignore

print(f"intercept: {model.intercept_}")
#intercept: 5.633333333333329

print(f"slope: {model.coef_}")
#slope: [0.54]
#abrar and me is learning the machine learning



