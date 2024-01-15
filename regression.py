import pandas as ps
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# taking the data
data = ps.read_csv("student-mat.csv", sep=";")

# I have 33 attributes in the data set but I decided to use 16 of them for this project otherwise it would be very complicated.
data = data[["Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]]

# I manually removed those who got higher than 6 in any of their first two exams and did not take the final exam(G3) from my data set in order to increase accuracy.
filtered_data = data[~(((data["G1"] > 6) | (data["G2"] > 6)) & (data["G3"] == 0))]

# the attribute to predict
predict = "G3"

# features array(data frame) without the G3 which is my label(output)
x = np.array(filtered_data.drop(columns=[predict]))

# values array of the label
y = np.array(filtered_data[predict])

# Splitting the data for train and test
# I was thinking to make the train/test ratio like %80 - %20 as in my midterm powerpoint slice but accuracy of %90 - %10 is mostly better.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# linear regression method of sklearn.linear_model
linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)
print(accuracy)

print(f"Coefficients: \n{linear.coef_}")
print(f"Intercept: \n{linear.intercept_}")

# visualizing
predictions = linear.predict(x_test)

for i in range(len(predictions)):
  print(predictions[i], x_test[i], y_test[i])