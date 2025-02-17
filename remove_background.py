from rembg import remove
from PIL import Image

# Charger l'image
input_path = "téléchargé (1).jpg"
output_path = "visage_sans_bg.png"

image = Image.open(input_path)

# Supprimer l'arrière-plan
output = remove(image)

# Sauvegarder l'image résultante
output.save(output_path)
