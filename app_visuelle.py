import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd
from qdrant_client import models  # <--- Ajouté comme demandé

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="IA Crédit Assistant", page_icon="🏦", layout="wide")

st.title("🏦 IA Crédit : Système d'Aide à la Décision")
st.markdown("Ce système compare le nouveau dossier à la **mémoire vectorielle** de la banque.")

# --- CHARGEMENT ---
@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = QdrantClient(path="./ma_memoire_qdrant")
    return model, client

try:
    model, client = load_resources()
    # st.success("✅ Cerveau et Mémoire chargés.") 
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# --- INITIALISATION SESSION (Obligatoire pour le Chatbot) ---
if "analyse_active" not in st.session_state:
    st.session_state.analyse_active = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- FORMULAIRE DE SAISIE ---
with st.sidebar:
    st.header("Nouveau Dossier")
    age = st.slider("Âge", 18, 500, 25)
    sexe = st.selectbox("Sexe", ["male", "female"])
    job = st.selectbox("Niveau Job (0=Non qualifié, 3=Cadre)", [0, 1, 2, 3])
    montant = st.number_input("Montant demandé (DT)", 0, 20000000000, 2000)
    duree = st.slider("Durée (mois)", 6, 72, 24)
    but = st.selectbox("But", ["car", "radio/TV", "furniture", "business", "education"])
    
    # MODIFICATION ICI : On utilise le session_state pour garder l'analyse active
    if st.button("🔍 Analyser le Risque"):
        st.session_state.analyse_active = True
        st.session_state.messages = [{"role": "assistant", "content": "Analyse terminée. Je suis prêt à répondre à vos questions sur ce dossier."}]

# --- LOGIQUE D'ANALYSE (S'active si le bouton a été cliqué) ---
if st.session_state.analyse_active:
    # 1. Reconstruire la phrase (Ton code)
    description = (
        f"Client de {age} ans, sexe {sexe}. "
        f"Demande de prêt de {montant} DT pour {but}. "
        f"Durée: {duree} mois. Job niveau {job}."
    )
    
    st.info(f"**Profil généré :** {description}")
    
    # 2. Vectorisation (Ton code)
    vecteur = model.encode(description).tolist()

    # Recherche compatible (ANN HNSW)
    try:
        reponse = client.search(
            collection_name="dossiers_clients",
            query_vector=vecteur,
            limit=5
        )
        resultats = reponse 
    except:
        reponse = client.query_points(
            collection_name="dossiers_clients",
            query=vecteur, 
            limit=5
        )
        resultats = reponse.points

    # 3. Initialisation des variables (Ton code)
    puissance_bad = 0.0
    puissance_good = 0.0
    scores = [] 

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Analyse des Voisins")
        
        if not resultats:
            st.warning("⚠️ Aucun profil similaire trouvé.")
        
        for hit in resultats:
            score = hit.score
            try:
                risque = hit.payload.get('Risk', hit.payload.get('risk_label', 'inconnu'))
            except:
                risque = "inconnu"

            scores.append(score)
            
            if risque == "bad":
                puissance_bad += score
                st.error(f"BAD (Similitude: {score:.1%})")
            else:
                puissance_good += score
                st.success(f"GOOD (Similitude: {score:.1%})")
                
            st.caption(f"ID Voisin: {hit.id}")

    # --- CALCULS AVANT AFFICHAGE (Ton code) ---
    if len(scores) > 0:
        moyenne_sim = sum(scores) / len(scores)
    else:
        moyenne_sim = 0

    # Règle 1 : L'IA trouve-t-elle que ça ressemble aux autres ?
    seuil_similarite_min = 0.65
    anomalie_ia = moyenne_sim < seuil_similarite_min

    # Règle 2 : Le montant est-il délirant ?
    seuil_montant_max = 200000000
    anomalie_montant = montant > seuil_montant_max

    # Flag Anomalie Global
    is_anomalie = anomalie_ia or anomalie_montant or age > 120 or montant > 200000000

    # Calcul Risque
    poids_bad_strict = puissance_bad * 3.0
    total_puissance = poids_bad_strict + puissance_good
    
    if total_puissance > 0:
        risque_pourcent = (poids_bad_strict / total_puissance) * 100
    else:
        risque_pourcent = 0

    # Seuil de refus
    seuil_refus = 25.0
    if risque_pourcent > seuil_refus:
        decision_finale = "REFUS"
    else:
        decision_finale = "ACCORD"


    # 4. COLONNE DE DROITE : MODIFICATION POUR ONGLETS ET CHATBOT
    with col2:
        # Création des onglets : 1 pour tes résultats, 1 pour le chat
        tab_decision, tab_chat = st.tabs(["📊 Décision & Analyse", "💬 Assistant IA"])

        # --- ONGLET 1 : TON AFFICHAGE HABITUEL ---
        with tab_decision:
            st.subheader("🤖 Décision IA (Politique Stricte)")

            # Affichage Anomalie (Ton code exact)
            if is_anomalie:
                st.error(f"⚠️ ALERTE ANOMALIE DÉTECTÉE")
                
                if anomalie_montant:
                    st.warning(f"💰 RAISON : Montant demandé ({montant} DT) suspect ou hors normes.")
                if anomalie_ia:
                    st.warning(f"🧠 RAISON : Le profil sémantique ne matche pas assez (Similitude: {moyenne_sim:.1%}).")
                
                st.info("🛑 Conseil : Ne pas traiter automatiquement. Escalade service fraude.")
                st.divider()

            # Affichage Risque (Ton code exact)
            if risque_pourcent > seuil_refus:
                couleur_barre = ":red["
            else:
                couleur_barre = ":green["

            st.markdown(f"Risque pondéré : **{couleur_barre}{risque_pourcent:.1f}%]** (Seuil: {seuil_refus}%)")
            st.progress(min(int(risque_pourcent), 100))

            # Courrier (Ton code exact)
            st.divider()
            st.subheader("📝 Projet de réponse")
            
            if decision_finale == "REFUS":
                st.error("REFUS RECOMMANDÉ")
                message = (
                    f"Monsieur/Madame,\n\n"
                    f"Nous ne pouvons donner suite.\n"
                    f"Politique stricte : Risque calculé de {risque_pourcent:.1f}% (Max autorisé: {seuil_refus}%).\n"
                    f"Similitude moyenne avec nos dossiers : {moyenne_sim:.1%}."
                )
            else:
                st.success("ACCORD DE PRINCIPE")
                message = (
                    f"Monsieur/Madame,\n\n"
                    f"Votre dossier est validé.\n"
                    f"Profil solide avec un risque estimé de seulement {risque_pourcent:.1f}%."
                )
                
            st.text_area("Brouillon :", value=message, height=200)

        # --- ONGLET 2 : LE CHATBOT (NOUVEL AJOUT) ---
        with tab_chat:
            st.subheader("💬 Discussion avec l'IA")
            
            # Afficher l'historique
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

            # Zone de saisie
            if prompt := st.chat_input("Posez une question sur ce dossier..."):
                # Ajouter message user
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)

                # Cerveau du Chatbot (Utilise tes variables)
                reponse_ia = ""
                txt_lower = prompt.lower()

                if "pourquoi" in txt_lower:
                    if is_anomalie:
                        reponse_ia = "Attention, c'est une ANOMALIE. Vérifiez le montant ou la cohérence du profil."
                    elif decision_finale == "REFUS":
                        reponse_ia = f"C'est refusé car le risque est de {risque_pourcent:.1f}%, ce qui dépasse la limite de {seuil_refus}%."
                    else:
                        reponse_ia = f"C'est accepté grâce à un risque faible de {risque_pourcent:.1f}%."
                elif "conseil" in txt_lower:
                    if decision_finale == "REFUS":
                        reponse_ia = "Essayez de baisser le montant ou d'allonger la durée."
                    else:
                        reponse_ia = "Le dossier est bon, rien à signaler."
                else:
                    reponse_ia = "Je peux vous expliquer le calcul du risque ou la raison du refus."

                # Ajouter réponse IA
                st.session_state.messages.append({"role": "assistant", "content": reponse_ia})
                st.chat_message("assistant").write(reponse_ia)
                st.rerun() # Force l'actualisation pour afficher la réponse





  