"""
Phone-side client placeholder (Termux/Python).
"""
import requests

r = requests.post(
    "http://10.206.66.228:8000",
    json={"msg": "hello"}
)

print(r.json())
