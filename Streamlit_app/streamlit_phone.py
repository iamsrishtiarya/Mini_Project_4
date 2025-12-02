import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Load models only
def load_model(path):
    with open(path, 'rb') as file:
        return joblib.load(file)

classifier = load_model(r"D:\GUVI-DS\Mini-Project5\best_model_random_forest.pkl")

# Define feature columns
features = [
    'E-commerce Spend (INR/month)', 'Spend_Ratio', 'Monthly Recharge Cost (INR)',
    'Data Usage per Hour', 'Data Usage (GB/month)', 'Usage per App',
    'Number of Apps Installed', 'Calls Duration (mins/day)', 'Call Duration per Day (hrs)',
    'Total_Usage_Time', 'Entertainment_Index', 'Streaming Time (hrs/day)',
    'Social Media Time (hrs/day)', 'Gaming Time (hrs/day)', 'Screen Time (hrs/day)',
    'OS_iOS', 'Age_Group_Middle Aged', 'Gender_Other', 'Age_Group_Adult', 'Gender_Male'
]

# Manual label mapping for predictions (encoded to original labels)
label_mapping = {
    0: "Education",
    1: "Entertainment",
    2: "Gaming",
    3: "Social Media",
    4: "Work",
}

st.set_page_config(page_title="Primary Use Predictor", layout="wide")
st.title("📱 Decoding Phone Usage Patterns in India")

tab1, tab2, tab3 = st.tabs(["📊 EDA Visuals", "🧠 Predict Primary Usage", "📊 User Clustering & Segmentation"])

# ---------------- Tab 1: EDA Visuals ----------------
with tab1:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Here are some insights into mobile usage behavior:")

    # Load dataset
    df = pd.read_csv(r"D:\GUVI-DS\Mini-Project5\phone_usage_india.csv")

    # ---  Distribution of App Usage Time by User Class ---
    st.subheader("📱 App Usage Time by User Class")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x='Social Media Time (hrs/day)', hue='Primary Use', kde=True, ax=ax)
    ax.set_title("Social Media Time Distribution by Primary Use")
    st.pyplot(fig)

    # ---  Feature Correlation Heatmap ---
    st.subheader("📉 Feature Correlation Heatmap")
    numerical_cols = df.select_dtypes(include='number')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numerical_cols.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix of Numeric Features")
    st.pyplot(fig)

    # ---  Screen Time by Primary Use (Box Plot) ---
    st.subheader("📦 Screen Time by User Class")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Primary Use', y='Screen Time (hrs/day)', ax=ax)
    ax.set_title("Screen Time Comparison Across User Classes")
    st.pyplot(fig)

# ---------------- Tab 2: Prediction ----------------
with tab2:
    st.header("🧠 Predict the Primary Use")

    with st.form("user_input_form"):
        input_data = {}

        col1, col2 = st.columns(2)
        with col1:
            input_data['E-commerce Spend (INR/month)'] = st.slider("E-commerce Spend", 0, 10000, 500)
            input_data['Monthly Recharge Cost (INR)'] = st.slider("Monthly Recharge Cost", 0, 1000, 200)
            input_data['Number of Apps Installed'] = st.slider("Number of Apps", 0, 100, 30)
            input_data['Calls Duration (mins/day)'] = st.slider("Calls Duration (mins/day)", 0, 300, 30)
            input_data['Data Usage (GB/month)'] = st.slider("Data Usage (GB/month)", 0, 100, 20)
            input_data['Streaming Time (hrs/day)'] = st.slider("Streaming Time", 0.0, 10.0, 1.0)

        with col2:
            input_data['Social Media Time (hrs/day)'] = st.slider("Social Media Time", 0.0, 10.0, 2.0)
            input_data['Gaming Time (hrs/day)'] = st.slider("Gaming Time", 0.0, 10.0, 1.0)
            input_data['Screen Time (hrs/day)'] = st.slider("Screen Time", 0.0, 24.0, 5.0)
            input_data['OS_iOS'] = 1 if st.selectbox("OS", ["Android", "iOS"]) == "iOS" else 0
            age_group = st.selectbox("Age Group", ["Teen", "Adult", "Middle Aged", "Senior"])
            input_data['Age_Group_Middle Aged'] = 1 if age_group == "Middle Aged" else 0
            input_data['Age_Group_Adult'] = 1 if age_group == "Adult" else 0
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            input_data['Gender_Male'] = 1 if gender == "Male" else 0
            input_data['Gender_Other'] = 1 if gender == "Other" else 0

        submitted = st.form_submit_button("Predict")
        if submitted:
            # 👉 Derive hidden features from input
            input_data['Call Duration per Day (hrs)'] = input_data['Calls Duration (mins/day)'] / 60
            input_data['Total_Usage_Time'] = (
                input_data['Screen Time (hrs/day)'] + input_data['Call Duration per Day (hrs)']
            )
            input_data['Usage per App'] = (
                input_data['Screen Time (hrs/day)'] / input_data['Number of Apps Installed']
            ) if input_data['Number of Apps Installed'] > 0 else 0

            input_data['Data Usage per Hour'] = (
                input_data['Data Usage (GB/month)'] / (30 * input_data['Total_Usage_Time'])
            ) if input_data['Total_Usage_Time'] > 0 else 0

            input_data['Entertainment_Index'] = (
                input_data['Gaming Time (hrs/day)'] +
                input_data['Streaming Time (hrs/day)'] +
                input_data['Social Media Time (hrs/day)']
            ) / input_data['Screen Time (hrs/day)'] if input_data['Screen Time (hrs/day)'] > 0 else 0

            input_data['Spend_Ratio'] = (
                (input_data['Monthly Recharge Cost (INR)'] + input_data['E-commerce Spend (INR/month)']) /
                input_data['Number of Apps Installed']
            ) if input_data['Number of Apps Installed'] > 0 else 0

            # Fill missing features with 0 to match model expectations
            for feat in features:
                if feat not in input_data:
                    input_data[feat] = 0

            # Ensure column order for prediction
            input_df = pd.DataFrame([input_data])
            input_df = input_df[features]
            prediction_encoded = classifier.predict(input_df)[0]
            prediction_label = label_mapping.get(prediction_encoded, "Unknown")
            st.success(f"✅ Predicted Primary Use: **{prediction_label}**")

# ---------------- Tab 2: Clustering ----------------
with tab3:
    st.header("📊🔍  User Clustering & Segmentation")

    # Load model
    kmeans = joblib.load(r"D:\GUVI-DS\Mini-Project5\model\kmeans_model.pkl")

    # Define the same features used during model training
    features = [
        'Screen Time (hrs/day)', 'Data Usage (GB/month)', 'Calls Duration (mins/day)',
        'Number of Apps Installed', 'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)',
        'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)', 'Monthly Recharge Cost (INR)'
    ]

    # Scale the dataset using freshly fit scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = df[features]
    X_scaled = scaler.fit_transform(X)

    # Assign clusters to the dataset (if not already present)
    if 'Cluster' not in df.columns:
        df['Cluster'] = kmeans.fit_predict(X_scaled)

    # --- User Input Section ---
    st.subheader("📱🧠 Input Your Mobile Usage Details")

    user_data = {}
    user_data['Screen Time (hrs/day)'] = st.slider("Screen Time (hrs/day)", 0.0, 24.0, 4.0)
    user_data['Data Usage (GB/month)'] = st.slider("Data Usage (GB/month)", 0.0, 100.0, 15.0)
    user_data['Calls Duration (mins/day)'] = st.slider("Calls Duration (mins/day)", 0, 300, 60)
    user_data['Number of Apps Installed'] = st.slider("Number of Apps Installed", 0, 100, 30)
    user_data['Social Media Time (hrs/day)'] = st.slider("Social Media Time (hrs/day)", 0.0, 10.0, 2.0)
    user_data['E-commerce Spend (INR/month)'] = st.slider("E-commerce Spend (INR/month)", 0, 20000, 500)
    user_data['Streaming Time (hrs/day)'] = st.slider("Streaming Time (hrs/day)", 0.0, 10.0, 1.5)
    user_data['Gaming Time (hrs/day)'] = st.slider("Gaming Time (hrs/day)", 0.0, 10.0, 1.0)
    user_data['Monthly Recharge Cost (INR)'] = st.slider("Monthly Recharge Cost (INR)", 0, 1000, 200)

    # Convert and scale input
    user_df = pd.DataFrame([user_data])
    user_scaled = scaler.transform(user_df)
    predicted_cluster = kmeans.predict(user_scaled)[0]

    st.success(f"✅ The user belongs to **Cluster {predicted_cluster}**")

    # Display user input details
    st.subheader("📋 Your Input Details")
    st.dataframe(user_df.T.rename(columns={0: "Value"}), use_container_width=True)

    # --- Cluster Visualization ---
    st.subheader("📍 Cluster Distribution")

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.scatterplot(
        x='Screen Time (hrs/day)',
        y='Data Usage (GB/month)',
        hue='Cluster',
        data=df,
        palette='tab10',
        s=40,
        ax=ax
    )

    # Mark user input point
    ax.scatter(
        user_data['Screen Time (hrs/day)'],
        user_data['Data Usage (GB/month)'],
        color='black',
        s=40,
        marker='X',
        label='User Input'
    )

    ax.set_title("User Segmentation\nScreen Time vs Data Usage", fontsize=10)
    ax.legend(loc='upper right', fontsize='small')  # Adjust legend size and location
    plt.tight_layout()
    st.pyplot(fig)

    # --- Create Cluster Profiles Table ---
    st.subheader("📊 Cluster Profiles")
    cluster_summary = df.groupby('Cluster')[features].mean()
    st.dataframe(cluster_summary, use_container_width=True)

    # --- Assign Names to Clusters ---
    cluster_info = {
        0: "🟢 Light users: Low spend, low screen/data usage",
        1: "🔵 Balanced users: Average behavior across metrics",
        2: "🟣 Entertainment-focused: High streaming/social/gaming time",
        3: "🔴 Power users: High data, apps, and screen time",
    }
    
    st.subheader("📌 Cluster Insights")
    st.markdown(f"Cluster {predicted_cluster}: **{cluster_info.get(predicted_cluster, 'No description available for this cluster.')}**")

    # --- Optional: Visualize Feature by Cluster ---
    st.subheader("📊 Feature Distribution by Cluster")
    
    # Feature visualization 
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fig, ax = plt.subplots(figsize=(3, 4))
        sns.boxplot(x='Cluster', y='Screen Time (hrs/day)', data=df, ax=ax)
        ax.set_title("Screen Time by Cluster", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(3, 4))
        sns.boxplot(x='Cluster', y='Social Media Time (hrs/day)', data=df, ax=ax)
        ax.set_title("Social Media Time by Cluster", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(3, 4))
        sns.boxplot(x='Cluster', y='E-commerce Spend (INR/month)', data=df, ax=ax)
        ax.set_title("E-commerce Spend by Cluster", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(3, 4))
        sns.boxplot(x='Cluster', y='Gaming Time (hrs/day)', data=df, ax=ax)
        ax.set_title("Gaming Time by Cluster", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    
