import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "/Users/dannychen/Downloads/student_performance_dataset.csv"
df = pd.read_csv(file_path)

# Set Streamlit app configuration
st.set_page_config(page_title="Student Performance Analysis", page_icon="📊", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Overview of the Data", "Exploratory Data Analysis (EDA)", "Machine Learning"])

# Home Page
if page == "Home":
    st.title("📊 Student Performance Analysis")
    st.image("https://images.unsplash.com/photo-1596495577886-d920f1a5014d", use_column_width=True)
    st.markdown("""
        Welcome to the **Student Performance Analysis** app! 🎓  
        This app explores a dataset containing information about students' academic performance, study habits, and external factors.  
        Use the sidebar to navigate between pages and discover insights about what influences final exam scores. 🚀  
    """)

# Overview of the Data
elif page == "Overview of the Data":
    st.title("📋 Overview of the Data")
    st.write("This section provides a brief overview of the dataset, including key features and data types.")
    
    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])

    st.subheader("Column Data Types")
    st.write(df.dtypes)

    st.subheader("Summary Statistics")
    st.write(df.describe())

# EDA Page
elif page == "Exploratory Data Analysis (EDA)":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.write("Visualizations to explore distributions and relationships within the data.")

    # Histogram for numerical features
    st.subheader("Distribution of Numerical Features")
    num_columns = ["Study_Hours_per_Week", "Attendance_Rate", "Past_Exam_Scores", "Final_Exam_Score"]
    for col in num_columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=20, kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Scatter plots for relationships between features
    st.subheader("Scatter Plots: Relationships between Features")
    scatter_features = [("Study_Hours_per_Week", "Final_Exam_Score"), ("Past_Exam_Scores", "Final_Exam_Score")]
    for x, y in scatter_features:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x, y=y, hue="Gender", palette="viridis", ax=ax)
        ax.set_title(f"{y} vs {x}")
        st.pyplot(fig)

    # Box plots for categorical features
    st.subheader("Box Plots: Categorical Features vs Final Exam Score")
    cat_columns = ["Parental_Education_Level", "Internet_Access_at_Home", "Extracurricular_Activities", "Gender"]
    for col in cat_columns:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=col, y="Final_Exam_Score", palette="Set2", ax=ax)
        ax.set_title(f"Final Exam Score by {col}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Bar charts for categorical data
    st.subheader("Bar Charts: Categorical Data Distribution")
    for col in cat_columns:
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', color='teal', ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Correlation Heatmap (Only on EDA Page)
    st.subheader("Correlation Heatmap")

    # Select only numerical columns
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = df[num_columns]

    # Compute the correlation matrix for numerical columns only
    corr_matrix = df_numeric.corr()

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax, fmt=".2f", linewidths=0.5)
    ax.set_title("Correlation Matrix of Features with Final Exam Score")
    st.pyplot(fig)

# Machine Learning Page
elif page == "Machine Learning":
    st.title("🤖 Machine Learning: Predicting Final Exam Score")
    st.write("""
        In this section, we will build a model to predict the **Final Exam Score** based on the available features. 
        We will use a **Linear Regression model** to do this.
    """)

    # Prepare the data
    st.subheader("Data Preparation")
    st.write("""
        We will use the following features to predict the final exam score:
        - Study Hours per Week
        - Attendance Rate
        - Past Exam Scores
        - Parental Education Level (encoded)
        - Internet Access at Home (encoded)
        - Extracurricular Activities (encoded)
        - Gender (encoded)
    """)

    # Encoding categorical variables (like 'Gender', 'Parental_Education_Level', etc.)
    df_encoded = df.copy()

    # Encode categorical variables (if applicable)
    df_encoded = pd.get_dummies(df_encoded, drop_first=True)

    # Check if Final_Exam_Score column exists in the dataset
    if "Final_Exam_Score" not in df_encoded.columns:
        st.error("Final_Exam_Score column not found in dataset. Please check the dataset.")
    else:
        # Define the features (X) and target (y)
        X = df_encoded.drop("Final_Exam_Score", axis=1)
        y = df_encoded["Final_Exam_Score"]

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize the Linear Regression model
        model = LinearRegression()

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate R² (Coefficient of Determination)
        r2 = r2_score(y_test, y_pred)

        # Display the R² score
        st.subheader(f"Model Performance: R² Score")
        st.write(f"The R² score for the model is: **{r2:.2f}**")

        # Explanation of how we calculated R²:
        st.write("""
            The **R² score** measures the proportion of the variance in the dependent variable (Final Exam Score) 
            that is predictable from the independent variables (features). An R² score close to 1 indicates that 
            the model is able to explain most of the variance in the data.
        """)

        # Optional: Display a scatter plot of actual vs predicted values
        st.subheader("Actual vs Predicted Scores")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
        ax.set_xlabel("Actual Final Exam Score")
        ax.set_ylabel("Predicted Final Exam Score")
        ax.set_title("Actual vs Predicted Final Exam Scores")
        st.pyplot(fig)
