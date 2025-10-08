import json
import os

# File to store variables
DATA_FILE = "./configData.json"

# Function to load a specific variable by name
def load_variable(name):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as file:
            variables = json.load(file)
            return variables.get(name, None)  # Return the value if exists, else None
    else:
        return None  # File doesn't exist

# Function to save or update a specific variable
def save_variable(name, value):
    variables = {}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as file:
            variables = json.load(file)
    variables[name] = value  # Update or add the variable
    with open(DATA_FILE, "w") as file:
        json.dump(variables, file, indent=4)