import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/extracted_features/hand_landmarks.csv')