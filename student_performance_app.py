{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e6b5c98a-d837-4c1b-aa72-bf21f37cabc5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-09 22:30:52.646 No runtime found, using MemoryCacheStorageManager\n",
      "2025-03-09 22:30:52.650 No runtime found, using MemoryCacheStorageManager\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import plotly.express as px\n",
    "\n",
    "# Load dataset\n",
    "@st.cache_data\n",
    "def load_data():\n",
    "    file_path = \"/Users/dannychen/Downloads/student_performance_dataset.csv\"\n",
    "    data = pd.read_csv(file_path)\n",
    "    return data\n",
    "\n",
    "data = load_data()\n",
    "\n",
    "# Set up Streamlit app\n",
    "st.set_page_config(page_title=\"Student Performance Analysis\", layout=\"wide\")\n",
    "\n",
    "# Home Page\n",
    "st.title(\"📊 Student Performance Analysis App\")\n",
    "st.markdown(\"Welcome to the Student Performance Analysis App! This app helps you explore factors affecting students' final exam scores through visualizations and predictive modeling.\")\n",
    "\n",
    "# Sidebar navigation\n",
    "menu = st.sidebar.selectbox(\"Select a Page\", (\"Home\", \"Overview of the Data\", \"EDA\"))\n",
    "\n",
    "if menu == \"Overview of the Data\":\n",
    "    st.header(\"Overview of the Data\")\n",
    "    st.write(\"Here's a quick look at the dataset:\")\n",
    "    st.dataframe(data.head())\n",
    "    st.write(\"Dataset Information:\")\n",
    "    st.text(data.info())\n",
    "    st.write(\"Summary Statistics:\")\n",
    "    st.write(data.describe())\n",
    "\n",
    "elif menu == \"EDA\":\n",
    "    st.header(\"Exploratory Data Analysis (EDA)\")\n",
    "    \n",
    "    # Correlation Heatmap\n",
    "    st.subheader(\"Correlation Heatmap\")\n",
    "    corr = data.corr()\n",
    "    fig, ax = plt.subplots()\n",
    "    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)\n",
    "    st.pyplot(fig)\n",
    "    \n",
    "    # Histograms\n",
    "    st.subheader(\"Distribution of Numerical Features\")\n",
    "    for col in ['Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores']:\n",
    "        fig, ax = plt.subplots()\n",
    "        sns.histplot(data[col], kde=True, ax=ax)\n",
    "        st.pyplot(fig)\n",
    "\n",
    "    # Scatter Plots\n",
    "    st.subheader(\"Relationships Between Features\")\n",
    "    fig = px.scatter(data, x='Study_Hours_per_Week', y='Final_Exam_Score', color='Gender', title='Study Hours vs Final Exam Score')\n",
    "    st.plotly_chart(fig)\n",
    "\n",
    "    # Box Plots\n",
    "    st.subheader(\"Distribution by Categorical Features\")\n",
    "    fig = px.box(data, x='Parental_Education_Level', y='Final_Exam_Score', color='Parental_Education_Level', title='Final Exam Scores by Parental Education Level')\n",
    "    st.plotly_chart(fig)\n",
    "\n",
    "    # Bar Charts\n",
    "    st.subheader(\"Impact of Internet Access and Extracurricular Activities\")\n",
    "    fig = px.bar(data, x='Internet_Access_at_Home', y='Final_Exam_Score', color='Internet_Access_at_Home', title='Internet Access vs Final Exam Score')\n",
    "    st.plotly_chart(fig)\n",
    "    fig = px.bar(data, x='Extracurricular_Activities', y='Final_Exam_Score', color='Extracurricular_Activities', title='Extracurricular Activities vs Final Exam Score')\n",
    "    st.plotly_chart(fig)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
