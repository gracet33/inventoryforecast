#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 04:48:54 2023

@author: GraceTee
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor


def train_and_predict_top_products(present_top_products, monthly_sales, product_index):
    num_predictions = 10
    product_index = product_index - 1
    filtered_sales = {}

    for index, row in present_top_products.iterrows():
        product_information = row['Product Information']
        filtered_sale = monthly_sales[monthly_sales['Product Information'] == product_information].copy()
        filtered_sale.rename(columns={'Quantity': "Sale"}, inplace=True)
        filtered_sale.drop('Product Information', axis=1, inplace=True)
        filtered_sale = filtered_sale.groupby('Order Date')['Sale'].sum().reset_index()
        filtered_sale.set_index("Order Date", inplace=True)

        key = index
        filtered_sales[key] = filtered_sale

    if product_index not in filtered_sales:
        raise KeyError(f"Product index {product_index} not found in filtered sales data.")

    filtered_sale = filtered_sales[product_index]
    filtered_sale['Sale_LastMonth'] = filtered_sale['Sale'].shift(1)
    filtered_sale['Sale_2Monthsback'] = filtered_sale['Sale'].shift(2)
    filtered_sale['Sale_3Monthsback'] = filtered_sale['Sale'].shift(3)
    filtered_sale = filtered_sale.dropna()

    x1, x2, x3, y = filtered_sale['Sale_LastMonth'].values, filtered_sale['Sale_2Monthsback'].values, \
                    filtered_sale['Sale_3Monthsback'].values, filtered_sale['Sale'].values
    final_x = np.column_stack((x1, x2, x3))
    X_train, X_test, y_train, y_test = final_x[:-num_predictions], final_x[-num_predictions:], \
                                       y[:-num_predictions], y[-num_predictions:]

    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    knn_pred = knn_model.predict(X_test)

    dates = filtered_sale.index[-num_predictions:]
    product_info = present_top_products.loc[product_index, 'Product Information']
    results_df = pd.DataFrame({'Date': dates, 'Predicted Sale': knn_pred, 'Actual Sale': y_test})

    return results_df


# Add your code to load the necessary data, e.g., present_top_products and monthly_sales
present_top_products = pd.read_csv("/Users/GraceTee/present_top_products.csv")  
monthly_sales = pd.read_csv("/Users/GraceTee/monthly_sales.csv")  

def main():
    st.title("Product Sales Prediction")

    product_index = st.selectbox("Select the ranking of product in the top five", range(1, 6))

    if st.button("Predict"):
        results = train_and_predict_top_products(present_top_products, monthly_sales, product_index)
        st.write(results)

if __name__ == "__main__":
    main()