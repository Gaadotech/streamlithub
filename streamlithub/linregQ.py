import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("OLS & LOOCV Regression")
uploaded_file=st.file_uploader("Upload CSV",type="csv")
X_axis = st.text_input("Enter X-axis name")
Y_axis = st.text_input("Enter Y-axis name")
tit = st.text_input("Enter Title name")

def mn(x):
    thesum=0
    for i in x:
        thesum=i+thesum
    mean=thesum/(len(x))
    return mean

if uploaded_file:
    df=pd.read_csv(uploaded_file)
    num_col=df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_col) <2:
        st.warning("Your CSV file is messed up, check it out")
    else:
        col1,col2=st.columns(2)
        x_var=col1.selectbox("Select X",num_col)
        y_var=col2.selectbox("Select Y",num_col)
        x=df[x_var].values
        y=df[y_var].values
        #Linear Reg & Q calc
        ssx=sum( (xi-mn(x))**2 for xi in x) #sum of sqaures for x
        sop= sum((xi-mn(x))*(yi-mn(y)) for xi,yi in zip(x,y)) #sum of products
        m=sop/ssx #slope
        b=mn(y)-mn(x)*m #intercept
        y_pred=[m*xi+b for xi in x]
        res=[yi-(m*xi+b) for xi,yi in zip(x,y) ] #residuals
        lev=[1/len(x)+ (xi-mn(x))/ssx for xi in x ] #leverage
        press=sum( (xi/(1-yi))**2 for xi,yi in zip(res,lev)) 
        ssy=sum( (yi-mn(y))**2 for yi in y) 
        ss_res = sum((yi - ypi)**2 for yi, ypi in zip(y, y_pred))
        q2=1-press/ssy
        r2=1-ss_res/ssy
        sig=(ss_res/(len(x)-2))**(0.5)
        ses=sig/(ssx**(0.5)) #standard error of slope
        sei=sig*((1/len(x)+mn(x)**2/ssx)**(0.5))
        fig, ax = plt.subplots()
        st.text("slope="+ str(m)+"+-"+str(ses))
        st.text("intercept="+str(b)+"+-"+str(sei))
        ax.scatter(x, y)
# Regression line
        show_regression = st.checkbox("Show regression line")
        if show_regression:
            ax.plot(x, y_pred, color="red")   # ← the line you’re toggling
        show_r2 = st.checkbox("Show R²")
        if show_r2:
            ax.text(0.80, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes,
            verticalalignment='top')
        show_q2=st.checkbox("Show Q²")
        if show_q2:
            ax.text(0.80, 0.90, f"Q² = {q2:.3f}", transform=ax.transAxes,
            verticalalignment='top')
        Ymx=st.checkbox("Show best fit equation")
        if Ymx:
            ax.text(0.705, 0.85, f"y = {b:.3f} + {m:.3f}x", transform=ax.transAxes,
            verticalalignment='top')
        ax.set_ylabel(Y_axis)
        ax.set_xlabel(X_axis)
        ax.set_title(tit)      
        st.pyplot(fig)
 
      