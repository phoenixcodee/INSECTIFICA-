import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# Page config
st.set_page_config(page_title="Insect Pest Classifier", layout="centered")
st.title("🦋 Insectifica ")
st.write("Upload an image of an insect pest to get the predicted species and details")

# Load class details from JSON
with open("pest.json", "r") as f:
    insect_data = json.load(f)

# List of class names for model prediction
class_names = [
    'Acanthophilus helianthi rossi',
    'Achaea janata',
    'Acherontia styx',
    'Adisura atkinsoni',
    'Aedes aegypti',
    'Aedes albopictus',
    'Agrotis ipsilon',
    'Alcidodes affaber',
    'Aleurodicus dispersus',
    'Amsacta albistriga',
    'Anarsia ephippias',
    'Anarsia epoitas',
    'Anisolabis stallii',
    'Antestia cruciata',
    'Aphis craccivora',
    'Apis mellifera',
    'Apriona cinerea',
    'Araecerus fasciculatus',
    'Atractomorpha crenulata',
    'Autographa nigrisigna',
    'Bagrada hilaris',
    'Basilepta fulvicorne',
    'Batocera rufomaculata',
    'Calathus erratus',
    'Camponotus consobrinus',
    'Chilasa clytia',
    'Chilo sacchariphagus indicus',
    'Conogethes punctiferalis',
    'Danaus plexippus',
    'Dendurus coarctatus',
    'Deudorix (Virachola) isocrates',
    'Elasmopalpus jasminophagus',
    'Euwallacea fornicatus',
    'Ferrisia virgata',
    'Formosina flavipes',
    'Gangara thyrsis',
    'Holotrichia serrata',
    'Hydrellia philippina',
    'Hypolixus truncatulus',
    'Leucopholis burmeisteri',
    'Libellula depressa',
    'Lucilia sericata',
    'Melanagromyza obtusa',
    'Mylabris phalerata',
    'Oryctes rhinoceros',
    'Paracoccus marginatus',
    'Paradisynus rostratus',
    'Parallelia algira',
    'Parasa lepida',
    'Pectinophora gossypiella',
    'Pelopidas mathias',
    'Pempherulus affinis',
    'Pentalonia nigronervosa',
    'peregrius maidis',
    'Pericallia ricini',
    'Perigea capensis',
    'Petrobia latens',
    'Phenacoccus solenopsis',
    'Phoetaliotes nebrascensis',
    'Phthorimaea operculella',
    'Phyllocnistis citrella',
    'Pieris brassicae',
    'Pulchriphyllium',
    'Rapala varuna',
    'Rastrococcus iceryoides',
    'Retithrips siriacus',
    'Retithrips syriacus',
    'Rhipiphorothrips cruentatus',
    'Rhopalosiphum maidis',
    'Rhopalosiphum padi',
    'Rhynchophorus ferrugineus',
    'Riptortus pedestris',
    'Sahyadrassus malabaricus',
    'Saissetia coffeae',
    'Streptanus aemulans',
    'sustama gremius',
    'Sylepta derogata',
    'Sympetrum signiferum',
    'Sympetrum vulgatum',
    'Tanymecus indicus Faust',
    'Tetraneura nigriabdominalis',
    'Tetrachynus cinnarinus',
    'Tetranychus piercei',
    'Thalassodes quadraria',
    'Thosea andamanica',
    'Thrips nigripilosus',
    'Thrips orientalis',
    'Thrips tabaci',
    'Thysanoplusia orichalcea',
    'Toxoptera odinae',
    'Trialeurodes rara',
    'Trialeurodes ricini',
    'Trichoplusia ni',
    'Tuta absoluta',
    'Udaspes folus',
    'Urentius hystricellus',
    'uroleucon carthami',
    'Vespula germanica',
    'Xeroma mura',
    'xylosadrus compactus',
    'Xylotrchus quadripes',
    'Zeuzera coffe',
    'non insects',
    'Papilio polytes',
    'Periplaneta americana'
]


# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mobilenetv2_insect_best.keras')
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an insect image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    def preprocess_image(img): 
       img = img.resize((224, 224)).convert("RGB")
       img_array = np.array(img, dtype=np.float32) / 255.0
       return np.expand_dims(img_array, axis=0)


    input_data = preprocess_image(image)

    # Predict
if uploaded_file is not None:
    with st.spinner("Classifying..."):
        predictions = model.predict(input_data)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])

    # ---------------- ERROR HANDLING ----------------
    if predicted_idx >= len(class_names):
        st.error("⚠️The provided image does not meet the classification requirements. Kindly upload another image for proper analysis.")
    else:
        predicted_class = class_names[predicted_idx]

        st.success(f"**Predicted Species:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2%}")

        # -------- JSON MATCHING SAFETY --------
        if predicted_class not in insect_data:
            st.warning(
                "Details for the predicted species are not available. "
                "Please upload another image or update the dataset."
            )
        else:
            details = insect_data[predicted_class]

            # ---------------- DISPLAY DETAILS ----------------
            st.write("## 🧬 Taxonomy")
            st.write(f"**Kingdom:** {details.get('Kingdom', 'N/A')}")
            st.write(f"**Phylum:** {details.get('Phylum', 'N/A')}")
            st.write(f"**Class:** {details.get('Class', 'N/A')}")
            st.write(f"**Order:** {details.get('Order', 'N/A')}")
            st.write(f"**Family:** {details.get('Family', 'N/A')}")
            st.write(f"**Genus:** {details.get('Genus', 'N/A')}")
            st.write(f"**Species:** {details.get('Species', 'N/A')}")

            st.write("## 🌿 Host Crops")
            st.write(details.get("Host Crops", "Not available"))

            st.write("## 🐛 Damage Symptoms")
            st.write(details.get("Damage Symptoms", "Not available"))

            st.write("## 🛡️ IPM Measures")
            st.write(details.get("IPM Measures", "Not available"))

            st.write("## ⚠️ Chemical Control")
            st.write(details.get("Chemical Control", "Not available"))

else:
    st.info("Please upload an image to get a prediction.")

st.markdown("---")
st.caption(
    "Ensure the uploaded image is clear and matches the trained insect classes. "
    "Model preprocessing must align with training settings."
)
