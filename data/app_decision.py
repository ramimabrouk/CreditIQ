from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import sys

# --- CONFIGURATION ---
SEUIL_ANOMALIE = 0.25

print("1️⃣ Chargement du cerveau et de la mémoire...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # On force la connexion locale au dossier que tu viens de créer
    client = QdrantClient(path="./ma_memoire_qdrant")
    print(f"   (Version Qdrant détectée : {client.__class__.__name__})")
except Exception as e:
    print(f"❌ Erreur au chargement : {e}")
    sys.exit()

# --- SCÉNARIO : UN NOUVEAU CLIENT ARRIVE ---
# Tu peux changer ce texte pour tester d'autres situations !
nouveau_dossier = "Client de 22 ans, femme. Demande 5000 DM pour une voiture. Pas d'épargne. Travaille depuis 1 an."
print(f"\n📄 Analyse du dossier : \"{nouveau_dossier}\"")

# 2. Vectorisation (Traduction en maths)
print("   -> Traduction en langage mathématique...")
vecteur_brut = model.encode(nouveau_dossier)
vecteur_client = vecteur_brut.tolist() # Important pour Qdrant

# 3. Recherche dans la mémoire
print("🔍 Recherche de cas similaires dans le passé...")

try:
    # On cherche les 5 dossiers les plus proches
    resultats = client.search(
        collection_name="dossiers_clients",
        query_vector=vecteur_client,
        limit=5
    )
except AttributeError:
    # Au cas où, méthode de secours
    resultats = client.query_points(
        collection_name="dossiers_clients",
        query=vecteur_client,
        limit=5
    ).points

# 4. Affichage des résultats
if not resultats:
    print("❌ Aucun résultat trouvé (La mémoire semble vide).")
    sys.exit()

votes_risques = []
scores_similarite = []

print("\n--- 📂 CAS HISTORIQUES SIMILAIRES (Explicabilité) ---")
for hit in resultats:
    score = hit.score
    risque = hit.payload['risk_label']
    description = hit.payload['description']
    
    scores_similarite.append(score)
    votes_risques.append(risque)
    
    # On met une icône rouge ou verte
    icone = "🔴" if risque == "bad" else "🟢"
    print(f"   {icone} Risque: {risque.upper()} (Similaire à {score*100:.1f}%) | {description[:60]}...")

# 5. DÉCISION FINALE
nb_bad = votes_risques.count("bad")
moyenne = sum(scores_similarite) / len(scores_similarite)

print("\n--- 🤖 RAPPORT DE DÉCISION ---")
if moyenne < SEUIL_ANOMALIE:
    print("⚠️ ANOMALIE DÉTECTÉE : Ce profil est trop bizarre (Inconnu au bataillon).")
    print("   -> Vérification humaine requise.")
elif nb_bad >= 3: # Majorité de mauvais payeurs
    print("❌ RECOMMANDATION : REFUS")
    print(f"   Raison : {nb_bad} des 5 profils similaires ont eu des défauts de paiement.")
else:
    print("✅ RECOMMANDATION : ACCORD")
    print(f"   Raison : La majorité des profils similaires ({5-nb_bad}/5) sont fiables.")