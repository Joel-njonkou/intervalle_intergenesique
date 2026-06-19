import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# CONFIGURATION
st.set_page_config(
    page_title="Prédiction Intervalle Intergénésique",
    page_icon="📈",
    layout="wide"
)


# CHARGEMENT DU MODÈLE
@st.cache_resource
def load_model():
    return joblib.load("modele_survie_intergenesique.pkl")

model = load_model()

# TITRE
st.title("📊 Prédiction de l'intervalle intergénésique")

st.markdown("""
Cette application estime la probabilité qu'une femme n'ait pas encore eu une nouvelle naissance
au fil du temps à partir d'un modèle de survie Random Survival Forest.
""")


# FORMULAIRE
with st.sidebar:

    st.header("Informations")

    milieu_residence = st.selectbox(
        "Milieu de résidence",
        ["Rural", "Urbain"]
    )

    niveau_instruction = st.selectbox(
        "Niveau d'instruction",
        [
            "Aucun",
            "Primaire",
            "Secondaire",
            "Supérieur"
        ]
    )

    indice_richesse = st.selectbox(
        "Indice de richesse",
        [
            "Très pauvre",
            "Pauvre",
            "Moyen",
            "Riche",
            "Très riche"
        ]
    )

    region = st.selectbox(
        "Région",
        [
            "Adamaoua",
            "Centre",
            "Est",
            "Extrême-Nord",
            "Littoral",
            "Nord",
            "Nord-Ouest",
            "Ouest",
            "Région_11",
            "Région_12",
            "Sud",
            "Sud-Ouest"
        ]
    )

    religion = st.selectbox(
        "Religion",
        [
            "Animiste",
            "Autre",
            "Catholique",
            "Islam",
            "Protestant",
            "Sans religion",
            "Inconnue"
        ]
    )

    sexe_enfant = st.selectbox(
        "Sexe de l'enfant précédent",
        [
            "Féminin",
            "Masculin"
        ]
    )

    survie_enfant_precedent = st.selectbox(
        "Survie de l'enfant précédent",
        [
            "Vivant",
            "Décédé"
        ]
    )

    age_premiere_naissance = st.selectbox(
        "Âge à la première naissance",
        [
            "<=15 ans",
            "16-20 ans",
            "21-25 ans",
            ">=26 ans"
        ]
    )

    instruction_mari = st.selectbox(
        "Instruction du mari",
        [
            "Aucun",
            "Primaire",
            "Secondaire",
            "Supérieur",
            "Inconnue"
        ]
    )

    predict_button = st.button(
        "Prédire"
    )


# PREDICTION
if predict_button:

    X_new = pd.DataFrame({
        "milieu_residence": [milieu_residence],
        "niveau_instruction": [niveau_instruction],
        "indice_richesse": [indice_richesse],
        "region": [region],
        "religion": [religion],
        "sexe_enfant": [sexe_enfant],
        "survie_enfant_precedent": [survie_enfant_precedent],
        "age_premiere_naissance": [age_premiere_naissance],
        "instruction_mari": [instruction_mari]
    })

    # Score de risque

    risk_score = model.predict(X_new)[0]

    st.subheader("Résultat")

    st.metric(
        "Score de risque",
        f"{risk_score:.2f}"
    )


    # COURBE DE SURVIE
    surv_fn = model.predict_survival_function(X_new)[0]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.step(
        surv_fn.x,
        surv_fn.y,
        where="post"
    )

    ax.set_title(
        "Courbe de survie prédite"
    )

    ax.set_xlabel(
        "Temps (mois)"
    )

    ax.set_ylabel(
        "Probabilité de ne pas avoir encore eu une nouvelle naissance"
    )

    ax.grid(True)

    st.pyplot(fig)

    # TABLEAU DES PROBABILITÉS
    horizons = [12, 24, 36, 48, 60]

    probs = []

    for t in horizons:

        probs.append(
            round(float(surv_fn(t)), 4)
        )

    result_df = pd.DataFrame({
        "Temps (mois)": horizons,
        "Probabilité de ne pas avoir une nouvelle naissance": probs
    })

    st.subheader(
        "Probabilités prédites"
    )

    st.dataframe(
        result_df,
        use_container_width=True
    )

    # INTERPRÉTATION
    p24 = float(surv_fn(24))

    st.subheader(
        "Interprétation"
    )

    st.info(
        f"""
        À 24 mois, le modèle estime une probabilité de {p24:.1%}
        de ne pas avoir encore eu une nouvelle naissance.

        La probabilité qu'une nouvelle naissance soit déjà survenue
        avant 24 mois est donc de {(1-p24):.1%}.
        """
    )