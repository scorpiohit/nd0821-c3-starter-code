import requests
import json

data = {"age": 45,
        "workclass": "Private",
        "fnlgt": 368561,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": " White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 55,
        "native-country": " United-States"}

response = requests.post("https://udacity-nd0821-c3-mohit.herokuapp.com/inference/", data=json.dumps(data))
print(response.status_code)
print(response.json())
