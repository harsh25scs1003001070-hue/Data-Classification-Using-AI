import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Data Classification",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🤖 Project 2: Data Classification Using AI")
st.markdown(
    """
    ### Supervised Learning Classification System

    This project demonstrates how an AI model learns from historical
    data and classifies new data into different categories.
    """
)

st.divider()

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = pd.Series(
    iris.target,
    name="target"
)

class_names = iris.target_names

# Create complete dataframe
df = X.copy()

df["target"] = y

df["species"] = df["target"].map(
    dict(enumerate(class_names))
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Model Settings")

test_size = st.sidebar.slider(
    "Testing Data (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)

max_depth = st.sidebar.slider(
    "Decision Tree Depth",
    min_value=1,
    max_value=10,
    value=3
)

# --------------------------------------------------
# DATASET SECTION
# --------------------------------------------------

st.header("1️⃣ Dataset")

st.write(
    """
    We are using the **Iris dataset**, a small and well-known
    classification dataset.
    """
)

st.dataframe(
    df.head(10),
    use_container_width=True
)

# Dataset statistics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Samples",
    len(df)
)

col2.metric(
    "Features",
    X.shape[1]
)

col3.metric(
    "Classes",
    len(class_names)
)

col4.metric(
    "Training Split",
    f"{100 - test_size}%"
)

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

st.subheader("Dataset Features")

feature_table = pd.DataFrame({
    "Feature": iris.feature_names,
    "Description": [
        "Length of sepal",
        "Width of sepal",
        "Length of petal",
        "Width of petal"
    ]
})

st.table(feature_table)

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

st.header("2️⃣ Train-Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100,
    random_state=42,
    stratify=y
)

col1, col2 = st.columns(2)

with col1:
    st.info(
        f"Training Samples: **{len(X_train)}**"
    )

with col2:
    st.info(
        f"Testing Samples: **{len(X_test)}**"
    )

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

st.header("3️⃣ AI Model Training")

model = DecisionTreeClassifier(
    max_depth=max_depth,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

st.success("✅ Decision Tree model trained successfully!")

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

st.metric(
    "🎯 Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.header("4️⃣ Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

image = ax.imshow(cm)

ax.set_title(
    "Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "Actual Class"
)

ax.set_xticks(
    range(len(class_names))
)

ax.set_yticks(
    range(len(class_names))
)

ax.set_xticklabels(
    class_names
)

ax.set_yticklabels(
    class_names
)

# Display values inside matrix
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

fig.colorbar(
    image,
    ax=ax
)

st.pyplot(fig)

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

st.header("5️⃣ Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True
)

# --------------------------------------------------
# USER PREDICTION
# --------------------------------------------------

st.header("6️⃣ 🔮 Try Your Own Prediction")

st.write(
    """
    Enter flower measurements below and let the trained
    AI model predict the flower species.
    """
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.8,
        step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=2.0,
        max_value=5.0,
        value=3.0,
        step=0.1
    )

with col3:
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=4.0,
        step=0.1
    )

with col4:
    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=3.0,
        value=1.2,
        step=0.1
    )

if st.button(
    "🚀 Predict Flower",
    use_container_width=True
):

    new_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(
        new_data
    )[0]

    probabilities = model.predict_proba(
        new_data
    )[0]

    predicted_class = class_names[
        prediction
    ]

    st.success(
        f"🌸 Predicted Flower: **{predicted_class.title()}**"
    )

    st.subheader(
        "Prediction Probability"
    )

    probability_df = pd.DataFrame({
        "Flower": [
            name.title()
            for name in class_names
        ],
        "Probability": [
            f"{probability * 100:.2f}%"
            for probability in probabilities
        ]
    })

    st.table(
        probability_df
    )

# --------------------------------------------------
# DECISION TREE
# --------------------------------------------------

st.header("7️⃣ How AI Makes Decisions")

st.write(
    """
    The Decision Tree learns patterns from the training data
    and creates decision rules automatically.
    """
)

fig2, ax2 = plt.subplots(
    figsize=(16, 8)
)

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    ax=ax2
)

st.pyplot(fig2)

# --------------------------------------------------
# PROCESS FLOW
# --------------------------------------------------

st.header("8️⃣ IPO Framework")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📥 INPUT")
    st.write(
        """
        • Iris dataset

        • Flower measurements

        • Training examples
        """
    )

with col2:
    st.subheader("⚙️ PROCESS")
    st.write(
        """
        • Data preprocessing

        • Train-test split

        • Decision Tree training

        • Pattern recognition
        """
    )

with col3:
    st.subheader("📤 OUTPUT")
    st.write(
        """
        • Predicted class

        • Accuracy

        • Confusion matrix

        • Classification report
        """
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Project 2: Data Classification Using AI | "
    "Supervised Machine Learning"
)