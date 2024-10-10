import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
import json
from sklearn.preprocessing import OneHotEncoder
import re
from lightgbm import LGBMClassifier

# Chargement du modèle MLflow
@st.cache_resource
def load_mlflow_model():
    # Remplacez le chemin par le chemin vers votre modèle MLflow
    model_path = "mlflow_model" 
    model = mlflow.pyfunc.load_model(model_path)
    return model

# Charger le masque de colonnes
@st.cache_resource
def load_column_mask():
    """Charge le masque des colonnes à partir du CSV de référence."""
    column_mask_df = pd.read_csv("mask.csv")
    return column_mask_df

# Charger le modèle et le masque des colonnes
model = load_mlflow_model()
column_mask_df = load_column_mask()

# Fonction de prétraitement (intégrant vos transformations)
def preprocess_data(df):
    """
    Applique les transformations de prétraitement sur le DataFrame d'entrée.

    Args:
    df (pd.DataFrame): Le DataFrame d'entrée avec les données brutes.

    Returns:
    pd.DataFrame: Le DataFrame prétraité.
    """
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], _ = pd.factorize(df[bin_feature])

    # Categorical features with One-Hot encode
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    # Instancie objet OHE
    onehot_encoder = OneHotEncoder(sparse_output=False)

    # Utilise OHE sur les colonnes catégorielles
    df_new_cols = onehot_encoder.fit_transform(df[categorical_columns])

    # Créez un DataFrame avec les nouvelles colonnes encodées
    encoded_df = pd.DataFrame(df_new_cols, columns=onehot_encoder.get_feature_names_out(categorical_columns))

    # Supprimer les colonnes catégorielles d'origine du DataFrame initial
    df = df.drop(categorical_columns, axis=1)

    # Concaténez le DataFrame d'origine avec le DataFrame contenant les nouvelles colonnes encodées
    df = pd.concat([df, encoded_df], axis=1)

    # Remplace les valeurs NaN dans la colonne `DAYS_EMPLOYED`
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # Créer de nouvelles features basées sur les colonnes existantes
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    df_zero = df.fillna(0)
    df = df_zero.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    
    # Aligner les colonnes du DataFrame de test avec celles du masque
    for col in column_mask_df.columns:
        if col not in df.columns:
            df[col] = 0  # Ajouter les colonnes manquantes avec des 0

    # Sélectionner les colonnes du masque pour obtenir le bon ordre
    df = df[column_mask_df.columns]
    
    # **Aligner les types de données avec le masque**
    for col in column_mask_df.columns:
        if col in df.columns:
            # Harmoniser le type de données avec le masque de référence
            df[col] = df[col].astype(column_mask_df[col].dtype)

    return df

# Interface utilisateur
st.title("Application de prédiction du défaut client")
st.write("Cette application utilise un modèle MLflow pour faire des prédictions à partir de fichiers CSV ou JSON.")

# Choisir le type de fichier (CSV ou JSON)
file_type = st.radio("Sélectionnez le type de fichier que vous souhaitez importer :", options=["CSV", "JSON"])

# Uploader le fichier correspondant
uploaded_file = st.file_uploader(f"Uploader un fichier {file_type}", type=[file_type.lower()])

if uploaded_file:
    if file_type == "CSV":
        # Lire le fichier CSV avec Pandas
        data = pd.read_csv(uploaded_file)
    elif file_type == "JSON":
        # Lire le fichier JSON avec Pandas
        data = pd.read_json(uploaded_file)
    
    # Affichage des données originales
    st.write("Données d'entrée :")
    st.write(data.head())
    st.write(data.shape)

    # Appliquer le prétraitement
    processed_data = preprocess_data(data)
    st.write("Données après prétraitement :")
    st.write(processed_data.head())
    st.write(processed_data.shape)

    # Vérifier que le modèle peut prédire avec ces données
    if st.button("Faire la Prédiction"):
        try:
            # Prédiction avec le modèle MLflow
            predictions = model.predict(processed_data)

            # Affichage des résultats de prédiction
            st.write("Résultats des prédictions :")
            st.write(predictions)

            # Ajouter les résultats au DataFrame pour un affichage combiné
            data['Prédiction'] = predictions
            st.write("Tableau des données avec les prédictions :")
            st.write(data)

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
else:
    st.write("Veuillez télécharger un fichier pour commencer les prédictions.")