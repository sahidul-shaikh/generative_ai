"""
Create an esmemble model based on the price prediction of three different agents-
1. Frontier agent
2. Specialist agent
3. Random forest agent
"""
# imports

from tqdm import tqdm
import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import chromadb
import pickle
# import agents
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.random_forest_agent import RandomForestAgent

# constants
DB = "products_vectorstore"

# Chromadb collection 
client = chromadb.PersistentClient(path=DB)
collection = client.get_or_create_collection('products')

# Load in the train pickle file:
with open('train.pkl', 'rb') as file:
    train = pickle.load(file)

# create instance for agents
specialist = SpecialistAgent()
frontier = FrontierAgent(collection)
random_forest = RandomForestAgent()


# utility function for product description
def description(item):
    return item.prompt.split("to the nearest dollar?\n\n")[1].split("\n\nPrice is $")[0]

# Pricing with three different agents and also get the actual price
specialists = []
frontiers = []
random_forests = []
prices = []
for item in tqdm(range(len(train))):
    text = description(item)
    specialists.append(specialist.price(text))
    frontiers.append(frontier.price(text))
    random_forests.append(random_forest.price(text))
    prices.append(item.price)

# Get minimum and  maximum price of three agents
mins = [min(s,f,r) for s,f,r in zip(specialists, frontiers, random_forests)]
maxes = [max(s,f,r) for s,f,r in zip(specialists, frontiers, random_forests)]

# create train dataframe 
X = pd.DataFrame({
    'Specialist': specialists,
    'Frontier': frontiers,
    'RandomForest': random_forests,
    'Min': mins,
    'Max': maxes,
})

# Convert actual price to y Series
y = pd.Series(prices)

# Train a Linear Regression
np.random.seed(42)

lr = LinearRegression()
lr.fit(X, y)

feature_columns = X.columns.tolist()

for feature, coef in zip(feature_columns, lr.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept={lr.intercept_:.2f}")

# Save the model
joblib.dump(lr, 'ensemble_model.pkl')