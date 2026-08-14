import requests
from pprint import pprint

"""
Documentation goes here for client
"""
"""
response = requests.get("http://localhost:8000")
print("Status code:", response.status_code)
print("Response:", response.json())


response = requests.get("http://localhost:8000/echo", params={"message": "Hello from the client!"})
print("Status code:", response.status_code)
print("Response:", response.json())
"""

response = requests.get("http://localhost:8000/predictions", params={"date": "2018-01-15"})
print("Status code:", response.status_code)
print("Response:")
pprint(response.json())
print(response.url)

response = requests.post(
    "http://localhost:8000/predictions",
    json={"date": "2018-05-15", "games": ["LAL @ MIN", "DEN @ MIL"]},
)
print("Status code:", response.status_code)
print("Response:")
pprint(response.json())
print(response.url)

# http://localhost:8000/predictions?date=YYYY-MM-DD

# TODO build out a proper if name main for this that you can use to customize an input function
