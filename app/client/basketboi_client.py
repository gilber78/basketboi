import requests
from pprint import pprint

"""
Documentation goes here for client
"""

response = requests.get("http://localhost:8000")
print("Status code:", response.status_code)
print("Response:")
pprint(response.json())
print(response.url)
print()

response = requests.get("http://localhost:8000/echo", params={"message": "Hello from the client!"})
print("Status code:", response.status_code)
print("Response:")
pprint(response.json())
print(response.url)
print()

response = requests.get("http://localhost:8000/predictions/", params={"date": "2018-01-15"})
print("Status code:", response.status_code)
print("Response:")
pprint(response.json())
print(response.url)
print()

response = requests.get("http://localhost:8000/predictions/", params={"date": "2018-05-15"})
print("Status code:", response.status_code)
print("Response:")
pprint(response.json())
print(response.url)
print()

response = requests.get(
    "http://localhost:8000/predictions/",
    params={"date": "2018-05-15", "games": ["LAL @ MIN", "DEN @ MIL"]},
)
print("Status code:", response.status_code)
print("Response:")
pprint(response.json())
print(response.url)

# http://localhost:8000/predictions?date=YYYY-MM-DD&games=AWY+%40+HME
