import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Revenue Predictor", layout="wide")
st.title("Restaurant Revenue Predictor")
st.caption("Upload a CSV file, explore a column, then fit a straight line to predict a target.")

#what is this tool for, and what happens when it is wrong?
st.info(
    "This model gives a rough estimate of a target such as monthly revenue from two inputs. "
    "A helpful starting view in the chart section could be: Column: Monthly Revenue, Chart Type: Scatter, Compare Against: "
    "Number of Customers. A helpful view in the Prediction section could be: Input 1: Number of customers, Input 2: Menu Price, "
    "Target: What you want to predict: Monthly Revenue. If the generated prediction is wrong a manager may over- or under-plan staff, "
    "stock and budget, it is important to treat this number as a starting point for planning, and not a set value."
)

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"{df.shape[0]:,} rows and {df.shape[1]} columns")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    #chart
    st.subheader("Chart")
    c1, c2, c3 = st.columns(3)
    chart_col = c1.selectbox("Column", df.columns.tolist())
    chart_type = c2.selectbox("Chart type", ["histogram", "scatter", "line"])

    if chart_type == "histogram":
        fig = px.histogram(df, x=chart_col, title=f"Distribution of {chart_col}")
    elif chart_type == "scatter":
        others = [c for c in numeric_cols if c != chart_col]
        if not others:
            st.error("Scatter needs a second numeric column.")
            st.stop()
        other = c3.selectbox("Compare against", others)
        fig = px.scatter(df, x=other, y=chart_col, title=f"{chart_col} versus {other}")
    else:
        fig = px.line(df, y=chart_col, title=f"{chart_col} by row")
    st.plotly_chart(fig, use_container_width=True)

    #regression with two inputs and one target
    st.subheader("Predict")
    if len(numeric_cols) < 3:
        st.error("Prediction needs at least three numeric columns (two inputs and a target).")
        st.stop()

    m1, m2, m3 = st.columns(3)
    x1 = m1.selectbox("Input 1", numeric_cols, index=0)
    x2_options = [c for c in numeric_cols if c != x1]
    x2 = m2.selectbox("Input 2", x2_options, index=0)
    y_options = [c for c in numeric_cols if c not in (x1, x2)]
    y_col = m3.selectbox("Target: what you want to predict", y_options)

    features = [x1, x2]
    data = df[features + [y_col]].dropna()
    if len(data) < 5:
        st.error("Need at least 5 rows with values in all three columns.")
        st.stop()

    X = data[features]
    y = data[y_col]
    model = LinearRegression()
    model.fit(X, y)

    r1, r2, r3 = st.columns(3)
    r1.metric(f"Each extra {x1} adds", f"{model.coef_[0]:,.2f}")
    r2.metric(f"Each extra {x2} adds", f"{model.coef_[1]:,.2f}")
    r3.metric("R2 on this data", f"{model.score(X, y):.2f}")

    st.markdown("**Try it**")
    v1, v2 = st.columns(2)
    value1 = v1.number_input(f"Enter a value for {x1}", value=float(data[x1].mean()))
    value2 = v2.number_input(f"Enter a value for {x2}", value=float(data[x2].mean()))

    new_row = pd.DataFrame([[value1, value2]], columns=features)
    prediction = model.predict(new_row)[0]
    st.metric(f"Predicted {y_col}", f"{prediction:,.0f}")

    for col, val in [(x1, value1), (x2, value2)]:
        low, high = data[col].min(), data[col].max()
        if val < low or val > high:
            st.warning(
                f"{col} = {val:,.2f} is outside the range the model learned from "
                f"({low:,.2f} to {high:,.2f}). Treat the number with caution."
            )
else:
    st.info("Upload a CSV file to get started. It needs at least three numeric columns.")
