import pickle
import sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import requests
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen as uReq

saved_model = pickle.load(open('linr_model.sav', 'rb'))
prediction = saved_model.predict([[ 1,1,1,1,1,1,1,1,1,1,1,1,1]])[0]

print(prediction)