from typing import TypedDict

class Person(TypedDict):
  name:str
  age:int
  
new_person = Person = {'name':"Shubham", 'age':43}

print(new_person)