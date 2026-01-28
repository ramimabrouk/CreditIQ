import sys
import subprocess

print(f"🔧 J'utilise le Python situé ici : {sys.executable}")
print("⏳ Début de l'installation forcée... Ne touche à rien.")

def forcer_installation(nom_paquet):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", nom_paquet])
        print(f"✅ {nom_paquet} installé avec succès.")
    except Exception as e:
        print(f"❌ Erreur sur {nom_paquet}: {e}")

# 1. On installe d'abord les bases manquantes
forcer_installation("setuptools")
forcer_installation("wheel")

# 2. On installe les outils de ton projet
forcer_installation("pandas")
forcer_installation("sentence-transformers")
forcer_installation("numpy")

print("\n🎉 TERMINÉ ! Ton environnement est prêt.")
print("Tu peux maintenant lancer ton script de vectorisation.")