from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data: X (feature), y (label)
house_size = np.array([[200], [450], [610], [770], [1000]])
price = np.array([40000, 52000, 75000, 76000, 90000])

# Create and train the model
model = LinearRegression()
model.fit(house_size, price)

# Predict
predictions = model.predict([[500]])
print(predictions)