import time
import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

st.title('Next Pitch ⚾️')

def app():
    import warnings
    warnings.filterwarnings('ignore')
    im = Image.open(
        "images/ben-sheets.jpg")
    st.image(im, width=400, caption='BY Marwah Faraj')

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    showWarningOnDirectExecution = True

    uploaded_file = st.file_uploader(
        "Upload your data to predict the next pitch", type=["csv"])
    if uploaded_file is not None:
        mlb_df = pd.read_csv(uploaded_file)

        # Display the data as a DataFrame
        st.dataframe(mlb_df)
    else:
        st.write("Invalid file. Please try again.")

    if st.button("Preprocess the data"):
        # progress_bar = st.progress(0)
        # for perc_completed in range(100):
        #     time.sleep(0.05)
        #     progress_bar.progress(perc_completed+1)
        with st.spinner('Wait for it...'):
            # pick the wanted features
            mlb_df = mlb_df[['inning', 'top', 'pcount_at_bat', 'pcount_pitcher', 'strikes', 'outs', 'start_tfs',
                            'pitcher_id', 'stand', 'b_height', 'p_throws', 'batter_id', 'start_tfs_zulu', 'pitch_type']]

            def time_segmentation(df, col):
                time_value_lst = []
                for i in df[col]:
                    if isinstance(i, str) and len(i.split()) > 1:
                        x = time.strptime(i.split()[1], '%H:%M:%S')
                        time_value_lst.append(datetime.timedelta(
                            hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds())
                    else:
                        time_value_lst.append(None)

                day_time_lst = []
                for time_val in time_value_lst:
                    if time_val is None:
                        day_time_lst.append('missing')
                    elif 32400 <= time_val <= 39600:
                        day_time_lst.append('morning')
                    elif 39600 <= time_val <= 54000:
                        day_time_lst.append('afternoon')
                    elif 54000 <= time_val < 64800:
                        day_time_lst.append('primetime')
                    elif 64800 <= time_val < 72000:
                        day_time_lst.append('evening')
                    else:
                        day_time_lst.append('other')

                return day_time_lst

            mlb_df['time'] = time_segmentation(mlb_df, 'start_tfs_zulu')

            # drop the extra columns
            mlb_df = mlb_df.drop('start_tfs_zulu', axis=1)

            # drop the null values in pitch_type
            mlb_df = mlb_df.dropna()

            # drop of unwanted pitch_type
            mlb_df = mlb_df[(mlb_df.pitch_type != 'UN') &
                            (mlb_df.pitch_type != 'AB')]

            # Balence the class using SMOTE

            # Select the categorical features to encode
            categorical_features = ['stand', 'b_height', 'p_throws', 'time']

            # Split the dataset into features (X) and target variable (y)
            X = mlb_df.drop('pitch_type', axis=1)
            y = mlb_df['pitch_type']

            # Encode the categorical features using OneHotEncoder
            encoder = OneHotEncoder()
            encoded_features = encoder.fit_transform(X[categorical_features])

            # Split the encoded features and remaining numerical features
            X_encoded = encoded_features.toarray()
            X_remaining = X.drop(categorical_features, axis=1)

            # Combine the encoded features with the remaining numerical features
            X_combined = np.hstack((X_encoded, X_remaining))

            # Encode the target variable using LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Determine the number of neighbors based on the minority class
            n_samples = len(y_encoded)
            n_minority = np.sum(y_encoded == np.min(y_encoded))
            n_neighbors = min(n_minority - 1, 5)

            # Apply SMOTE to oversample the minority class
            smote = SMOTE(sampling_strategy='not majority',
                          k_neighbors=n_neighbors)
            X_resampled, y_resampled = smote.fit_resample(
                X_combined, y_encoded)

            # Sclae the data
            scaler = RobustScaler()
            X_resampled_scaled = scaler.fit_transform(X_resampled)

            # the following steps should not be there in the future
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled_scaled, y_resampled, test_size=0.2, random_state=42)

        st.write("Data preprocessed, ready to predict the next pitch")
        time.sleep(0.05)
        st.write("Prediction in progress")

        with st.spinner('Wait for it...'):

            filename = 'saved_model/extratreeclf.pkl'
            loaded_model = joblib.load(filename)
            unseen_data = X_test[1].reshape(1, -1)
            pred = loaded_model.predict(unseen_data)
            # Assuming label_encoder is your LabelEncoder object
            y_pred_labels = label_encoder.inverse_transform(pred)
            y_true_lables = label_encoder.inverse_transform(y_test[1:2])

        st.write(
            f"Your next pitch type is {y_pred_labels} as the model predicti it, and the actuals is {y_true_lables}")


if __name__ == '__main__':
    app()
