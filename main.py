import streamlit as st
import pandas as pd
import numpy as np
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split

st.title('No Code ML Models 🤖')
st.header('An easy Drag-n-Drop dashboard to build Machine Learning Models')


uploaded_file = st.file_uploader("Drop your CSV or XLSX file here:", type=['csv' , 'xlsx'], accept_multiple_files=False)

if uploaded_file != None:

    try:
        df = pd.read_csv(uploaded_file)
    except:
        df = pd.read_excel(uploaded_file)

    if st.button('Click here to view your datafile'):
        st.dataframe(df)
    columns = list(df.columns)

    #st.button(columns)
    delete_columns = st.multiselect(
    "Select the columns you'd like to delete (You can leave this empty too 😊)",
    columns)

    df = df.drop(columns=delete_columns, axis = 1)

    ind_variables = st.multiselect(
    "Select your Independent Variables (By default, we take all of them 😎)",
    list(df.columns))

    target = st.radio(
    "Select your Target Variable 🎯",
    list(df.columns))

    st.write('You selected:', target)

    # Creating a new dataframe using the user mentioned independent variables
    new_df = pd.DataFrame()


    def best_model():
        i = 0
        while i < len(ind_variables):
            try:
                ind_var_arr = np.vstack([df[ind_variables[i]],df[ind_variables[i-1]]])
            except:
                pass
            i += 1

        y = (df[target].to_numpy())
        X = np.transpose(ind_var_arr)
        st.text(y.shape)
        st.text(X.shape)


        X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.2, random_state = 64)
        st.text(X_train)
        st.text(y_train)
        st.text(X_test)
        st.text(y_test)
        
        reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)

        model, predict = reg.fit(X_train, X_test, y_train, y_test)



        #y = (df.iloc[: , -1:]).to_numpy()
    best_model()

    #if st.button('Click here to see your edited datafile'):
    #    st.dataframe(df)


