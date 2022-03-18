import pandas
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def input_score():
        status = True
        while status:
             m= input('Math score - ( 0 - 100 ) > ')
             s = input('Culture score - ( 0 - 100 ) > ')
             a = input('Art score - ( 0 - 100 ) > ')
             if not m.isnumeric() or not s.isnumeric() or not s.isnumeric():
                 status = True
                 print('invalid input try again')
             else:
                 math = int(m)
                 social = int(s)
                 art = int(a)
                 if math>100 or math<0 or social>100 or social<0 or art>100 or art<0:
                     status = True
                     print('invalid input try again')
                 else:
                     return math, social, art
out = input_score()
math_score = out[0]
social_score = out[1]
art_score = out[2]
orient = {0: 'Enginer', 1: 'Social', 3: 'Art'}
scale = StandardScaler()
warnings.simplefilter(action='ignore')
df = pandas.read_csv("data/train.csv")
df = df.drop(columns = ["Name"])
X = df[['Math', 'Language', 'Art']]
y = df['orientation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
print(score)
predicted = model.predict([[math_score, social_score, art_score]])
pre = int(predicted)
print(predicted)
print(f"This is what we recommend for you: {orient[pre]}")