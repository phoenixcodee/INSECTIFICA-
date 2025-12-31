import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
# Hide hamburger menu, footer, header, and toolbar completely
# Hide Streamlit's top-right menu (hamburger ☰), Manage app, Deploy button, toolbar, footer, etc.
st.markdown("""
<style>
[data-testid="stToolbar"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

hide_streamlit_elements = """
    <style>
    /* Hide the main hamburger menu */
    #MainMenu {visibility: hidden !important;}
    
    /* Hide the toolbar (new in recent versions) */
    .stToolbar {display: none !important;}
    
    /* Hide Deploy button */
    .stDeployButton {display: none !important;}
    
    /* Hide header bar */
    header {visibility: hidden !important;}
    
    /* Hide footer ("Made with Streamlit") */
    footer {visibility: hidden !important;}
    
    /* Extra safety for any new elements */
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="manage-app-button"] {display: none !important;}
    </style>
"""
st.markdown(hide_streamlit_elements, unsafe_allow_html=True)

# Optional: make the page look even cleaner


st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="INSECTIFICA",
    page_icon="🐞",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>

/* --------------------------------------------------
   GLOBAL APP THEME
-------------------------------------------------- */
.stApp {
    background: linear-gradient(139deg, #f6fff8, #e8f5e9);
    color: #1b5e20;
    font-family: "Segoe UI", "Roboto", sans-serif;
}

/* --------------------------------------------------
   HEADINGS
-------------------------------------------------- */
h1, h2, h3, h4, h5, h6 {
    text-align: center;
    color: #1b5e20 !important;
    font-weight: 700;
}

/* --------------------------------------------------
   BUTTON STYLING
-------------------------------------------------- */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    background: linear-gradient(135deg, #2e7d32, #66bb6a);
    color: white !important;
    font-size: 18px;
    font-weight: 600;
    padding: 0.75em;
    border: none;
    transition: all 0.2s ease-in-out;
}

/* Hover effect */
.stButton > button:hover {
    background: linear-gradient(135deg, #388e3c, #81c784);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    transform: translateY(-1px);
}

/* Active (click) effect */
.stButton > button:active {
    background: linear-gradient(135deg, #1b5e20, #43a047) !important;
    transform: scale(0.97);
}

/* Disabled / loading look */
button[disabled] {
    background: linear-gradient(135deg, #1b5e20, #66bb6a) !important;
    opacity: 0.75;
    cursor: wait;
}

/* --------------------------------------------------
   FILE UPLOADER
-------------------------------------------------- */
[data-testid="stFileUploader"] {
    border: 2px dashed #2e7d32;
    border-radius: 16px;
    padding: 1em;
    background-color: #f1f8e9;
}

/* --------------------------------------------------
   IMAGES
-------------------------------------------------- */
img {
    border-radius: 16px;
    max-width: 100%;
}

/* --------------------------------------------------
   CARD UI
-------------------------------------------------- */
.card {
    background: white;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 22px;
    color: #1b5e20;
}

/* --------------------------------------------------
   FOOTER
-------------------------------------------------- */
.footer {
    text-align: center;
    font-size: 13px;
    color: #2e7d32 !important;
    margin-top: 30px;
}

/* --------------------------------------------------
   MOBILE RESPONSIVENESS
-------------------------------------------------- */
@media (max-width: 768px) {
    h1 {
        font-size: 34px !important;
    }

    .stButton > button {
        font-size: 16px;
        padding: 0.65em;
    }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load Data & Model
# --------------------------------------------------
with open("pest.json", "r") as f:
    insect_data = json.load(f)

class_names = [
    'Acanthophilus helianthi rossi', 'Achaea janata', 'Acherontia styx', 'Adisura atkinsoni',
    'Aedes aegypti', 'Aedes albopictus', 'Agrotis ipsilon', 'Alcidodes affaber',
    'Aleurodicus dispersus', 'Amsacta albistriga', 'Anarsia ephippias', 'Anarsia epoitas',
    'Anisolabis stallii', 'Antestia cruciata', 'Aphis craccivora', 'Apis mellifera',
    'Apriona cinerea', 'Araecerus fasciculatus', 'Atractomorpha crenulata', 'Autographa nigrisigna',
    'Bagrada hilaris', 'Basilepta fulvicorne', 'Batocera rufomaculata', 'Calathus erratus',
    'Camponotus consobrinus', 'Chilasa clytia', 'Chilo sacchariphagus indicus',
    'Conogethes punctiferalis', 'Danaus plexippus', 'Dendurus coarctatus',
    'Deudorix (Virachola) isocrates', 'Elasmopalpus jasminophagus', 'Euwallacea fornicatus',
    'Ferrisia virgata', 'Formosina flavipes', 'Gangara thyrsis', 'Holotrichia serrata',
    'Hydrellia philippina', 'Hypolixus truncatulus', 'Leucopholis burmeisteri',
    'Libellula depressa', 'Lucilia sericata', 'Melanagromyza obtusa', 'Mylabris phalerata',
    'Oryctes rhinoceros', 'Paracoccus marginatus', 'Paradisynus rostratus', 'Parallelia algira',
    'Parasa lepida', 'Pectinophora gossypiella', 'Pelopidas mathias', 'Pempherulus affinis',
    'Pentalonia nigronervosa', 'peregrius maidis', 'Pericallia ricini', 'Perigea capensis',
    'Petrobia latens', 'Phenacoccus solenopsis', 'Phoetaliotes nebrascensis',
    'Phthorimaea operculella', 'Phyllocnistis citrella', 'Pieris brassicae', 'Pulchriphyllium',
    'Rapala varuna', 'Rastrococcus iceryoides', 'Retithrips siriacus', 'Retithrips syriacus',
    'Rhipiphorothrips cruentatus', 'Rhopalosiphum maidis', 'Rhopalosiphum padi',
    'Rhynchophorus ferrugineus', 'Riptortus pedestris', 'Sahyadrassus malabaricus',
    'Saissetia coffeae', 'Streptanus aemulans', 'sustama gremius', 'Sylepta derogata',
    'Sympetrum signiferum', 'Sympetrum vulgatum', 'Tanymecus indicus Faust',
    'Tetraneura nigriabdominalis', 'Tetrachynus cinnarinus', 'Tetranychus piercei',
    'Thalassodes quadraria', 'Thosea andamanica', 'Thrips nigripilosus', 'Thrips orientalis',
    'Thrips tabaci', 'Thysanoplusia orichalcea', 'Toxoptera odinae', 'Trialeurodes rara',
    'Trialeurodes ricini', 'Trichoplusia ni', 'Tuta absoluta', 'Udaspes folus',
    'Urentius hystricellus', 'uroleucon carthami', 'Vespula germanica', 'Xeroma mura',
    'xylosadrus compactus', 'Xylotrchus quadripes', 'Zeuzera coffe', 'non insects',
    'Papilio polytes', 'Periplaneta americana'
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mobilenetv2_insect.tflite')

model = load_model()

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def ui_card(title, content):
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
            {content}
        </div>
        """, unsafe_allow_html=True
    )

def how_it_works_section():
    ui_card(
        "🧠 How Insectifica Works",
        """
        <b>📸 Step 1: Snap or Upload a Photo</b><br>
        Use your device camera to take a clear, focused photo of the insect or pest,
        or upload an image from your gallery.<br><br>
        <b>🤖 Step 2: AI-Powered Analysis</b><br>
        Insectifica’s deep learning model analyzes the image by comparing it with a
        large entomological database, focusing on:
        <ul>
            <li>Body shape & size</li>
            <li>Color patterns</li>
            <li>Wing structure</li>
            <li>Antennae & leg features</li>
        </ul>
        <b>🐞 Step 3: Identification & Insights</b><br>
        Within seconds, the app provides:
        <ul>
            <li>Common & Scientific Name</li>
            <li>Taxonomic Classification</li>
            <li>Behaviour & Habitat</li>
            <li>Ecological Role (Pest / Beneficial / Neutral)</li>
        </ul>
        """
    )
    st.info("💡 Tip: For best accuracy, ensure the insect is well-lit and clearly visible.")

# --------------------------------------------------
# Page Definitions
# --------------------------------------------------
def intro_page():
    st.title("🐞 INSECTIFICA 🔍")
    st.subheader("AI-Powered Insect & Pest Identification")
    st.markdown("""
    **Insectifica** helps identify insects and pests instantly using artificial intelligence
    and image recognition. Designed for **students, farmers, researchers, and nature enthusiasts**.
    """)
    st.divider()
    how_it_works_section()
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("🔍 Start Identification", use_container_width=True):
            with st.spinner("Wait Loading..."):
              st.session_state.page = "classification"
              st.rerun()
    st.divider()

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        if st.button("ℹ️ About App"):
            with st.spinner("Wait Loading..."):
              st.session_state.page = "about_app"
              st.rerun()
    with col_d:
        if st.button("👨‍🔬 Developers"):
            with st.spinner("Wait Loading..."):
                 st.session_state.page = "developers"
                 st.rerun()

def about_app_page():
     st.title("ℹ️ About INSECTIFICA")

     st.markdown("""
        **Insectifica** is an AI-powered mobile application designed to help users instantly identify
        insects, pests, and other arthropods from photographs. It leverages advanced image recognition
        techniques and a comprehensive entomological database to make insect identification accessible
        to professionals, scientists, gardeners, farmers, and nature enthusiasts alike.

        Insectifica is an **educational and research-support application** developed by the  
        **Department of Biotechnology, St. Joseph’s College (Autonomous), Tiruchirappalli**.

        Developed with a commitment to educational and research excellence, Insectifica reflects
        St. Joseph’s College and the Department of Biotechnology’s ongoing mission to promote
        scientific awareness, support research, and create innovative tools that empower learners
        and professionals in the field of Biotechnology.
        """
    )

     st.divider()

     st.header("🎯 Core Purpose")
     st.markdown(
        """
        Insectifica’s primary goal is to provide **fast and accurate identification**
        of insects and pests using a simple photograph captured through a smartphone camera.

        Whether encountering a tiny beetle in a home garden, a mysterious insect indoors,
        or a potentially harmful pest in agricultural fields, Insectifica delivers
        **reliable identification results** along with **educational insights**—all with
        minimal effort.
        """
    )

     st.divider()
     if st.button("➡️ Features & Use Cases"):
         with st.spinner("Wait Loading..."):
          st.session_state.page = "features"
          st.rerun()
     if st.button("⬅️ Back"):
         with st.spinner("Wait Loading..."):
          st.session_state.page = "intro"
          st.rerun()

def features_page():
     st.title("✨ Features & Use Cases")
 
     st.header("🔑 Key Features of Insectifica")
     st.markdown(
        """
        • **Instant Identification:**  
        Identify insects and arthropods instantly from photographs using advanced
        machine learning—ideal for both casual users and experts.

        • **Comprehensive Species Database:**  
        Access detailed profiles of hundreds of insect and pest species including
        butterflies, ants, beetles, moths, spiders, and major agricultural pests.

        • **Pest vs. Beneficial Indicator:**  
        Clearly distinguish whether a species is harmful (pest), neutral, or beneficial
        (such as pollinators and natural predators).

        • **Habitat & Behaviour Insights:**  
        Each identification includes habitat preferences, life cycle details, feeding
        habits, and ecological roles.

        • **Identification History:**  
        Save and review past identifications—useful for students, educators, researchers,
        and biodiversity documentation.

        • **Community & Sharing:**  
        Share discoveries with peers or within a community to encourage collaborative
        learning and nature awareness.
        """
    )

     st.divider()

     st.header("👥 Use Cases")
     st.markdown(
        """
        • **Gardeners & Homeowners:**  
        Identify pests affecting plants and learn natural pest management tips.

        • **Students & Educators:**  
        Use real identifications in biology classes and field projects for hands-on learning.

        • **Farmers & Agriculturists:**  
        Spot agricultural pests early and decide integrated pest management steps.

        • **Nature Enthusiasts:**  
         Explore biodiversity around you and build personal insect sighting collections.
        """
    )

     st.divider()

     st.header("🌍 Why Insectifica Is Useful")
     st.markdown(
        """
        Insectifica bridges the gap between expert entomological identification and everyday curiosity. By combining AI technology with scientific databases, it transforms insect and pest encounters into educational moments, helps reduce fear or misinformation about bugs and enables data collection for broader ecological insights..
        """
    )

     st.divider()

     st.header("📸 Notes & Best Practices")
     st.markdown(
        """
        • Capture clear, well-focused images under good lighting conditions.  
        • Take photographs from multiple angles whenever possible.  
        • Ensure key anatomical features such as wings, legs, antennae, and body patterns
          are clearly visible to improve identification accuracy.
        """
    )

     st.divider()

     st.header("📸 Best Practices")
     st.markdown("""
      • Accuracy improves with clear, focused photos taken from multiple angles — close enough to see key insect traits.
      """)
     if st.button("👨‍🔬 Developers"):
          with st.spinner("Wait Loading..."):
            st.session_state.page = "developers"
            st.rerun()
     if st.button("⬅️ Back"):
          with st.spinner("Wait Loading..."):
            st.session_state.page = "about_app"
            st.rerun()

def developers_page():
     st.title("👨‍🔬 Development Team")

     st.markdown("""
    **Department of Biotechnology**  
    St. Joseph’s College (Autonomous)  
    Tiruchirappalli – 620 002  
    Contact mail id:  
    edward_bt2@mail.sjctni.edu  
    cisgene.edward@gmail.com
    """)

     st.divider()

     st.markdown("""
    **App Concept & Design**  
    Dr. A. Edward  

    **Development & Programming**  
    Dr. A. Edward  
    Dr. V. Swabna  
    Dr. A. Asha Monica  
    Dr. Pavulraj Michael SJ

    **Scientific Data Verification**  
    Dr. V. Swabna, Dr. A. Asha Monica and Dr. Pavulraj Michael SJ

    **Guidance & Supervision**  
    Dr. Pavulraj Michael SJ  
    Rector, St. Joseph’s College (Autonomous)  
    Tiruchirappalli – 620 002
    """)

     if st.button("⬅️ Back to Home"):
         with st.spinner("Wait Loading..."):
              st.session_state.page = "intro"
              st.rerun()


def classification_page():
    st.title("🔍 Insect Identification")
    
    # ---------------- Professional Header Card ----------------
    st.markdown("""
    <div class="card" style="text-align: center; padding: 20px; margin-bottom: 30px;">
        <h2 style="color: #2e7d32; margin-bottom: 10px;">📸 Upload or Snap a Photo ⬇️</h2>
        <p style="font-size: 16px; color: #1b5e20;">
            Take a clear photo of the insect or upload from your gallery for instant AI-powered identification.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---------------- Centered File Uploader ----------------
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="Supported: JPG, PNG | Max size: 10MB"
        )
    
    # ---------------- Photo Tips Section (Always Visible) ----------------
    st.markdown("""
    <div style="background: #f1f8e9; border-radius: 16px; padding: 20px; margin-bottom: 25px; border-left: 5px solid #4caf50;">
        <h4 style="color: #2e7d32; text-align: center;">💡 Best Tips for Accurate Results</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 15px;">
            <div style="text-align: center;">
                <div style="font-size: 40px; margin-bottom: 8px;">📸</div>
                <b>Clear & Focused</b><br>
                <small>Get close, keep the insect sharp</small>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 40px; margin-bottom: 8px;">☀️</div>
                <b>Natural Light</b><br>
                <small>Avoid shadows, use daylight</small>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 40px; margin-bottom: 8px;">👀</div>
                <b>Multiple Angles</b><br>
                <small>Side, top, wings if visible</small>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 40px; margin-bottom: 8px;">👐</div>
                <b>Plain Background</b><br>
                <small>Leaf, wall, or hand works best</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ---------------- Image Processing (Only if uploaded) ----------------
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
       
        # Display uploaded image beautifully
        st.markdown("<h3 style='text-align: center; color: #2e7d32;'>Uploaded Image</h3>", unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="Ready for analysis")
        
        # Preprocess and predict
        img = image.resize((190, 190))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        with st.spinner("🤖 AI is analyzing the insect... Please wait a moment"):
            predictions = model.predict(img_array)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))  # Safest way: get max prob as clean float
        
        st.markdown("---")
        
        if predicted_idx >= len(class_names):
            st.error("⚠️ Unable to classify. Please try a clearer image of a single insect.")
        else:
            predicted_class = class_names[predicted_idx]
            # Confidence bar with animation feel
            st.success(f"**Identified Species:** {predicted_class}")
            st.progress(confidence)
            st.write(f"**Confidence Level:** {confidence:.1%}")
            
            # Detailed Info
            if predicted_class in insect_data:
                details = insect_data[predicted_class]
                st.markdown("## 🧬 Taxonomic Classification")
                col_k, col_p, col_c = st.columns(3)
                with col_k: st.write(f"**Kingdom:** {details.get('Kingdom', 'N/A')}")
                with col_k: st.write(f"**Phylum:** {details.get('Phylum', 'N/A')}")
                with col_k: st.write(f"**Class:** {details.get('Class', 'N/A')}")
                
                col_o, col_f = st.columns(2)
                with col_o: st.write(f"**Order:** {details.get('Order', 'N/A')}")
                with col_o: st.write(f"**Family:** {details.get('Family', 'N/A')}")
                
                st.write(f"**Genus:** {details.get('Genus', 'N/A')}")
                st.write(f"**Species:** {details.get('Species', 'N/A')}")
                
                st.markdown("## 🌿 Host Crops")
                st.info(details.get("Host Crops", "Not available"))
                
                st.markdown("## 🐛 Damage Symptoms")
                st.warning(details.get("Damage Symptoms", "Not available"))
                
                st.markdown("## 🛡️ Integrated Pest Management (IPM)")
                st.success(details.get("IPM Measures", "Not available"))
                
                st.markdown("## ⚠️ Chemical Control (If Needed)")
                st.error(details.get("Chemical Control", "Not available"))
            else:
                st.warning("🔍 Detailed information for this species is not yet available in our database.")
        
        # Back Button after results
        st.markdown("---")
        col_back1, col_back2, col_back3 = st.columns([1, 1, 1])
        with col_back2:
            if st.button("⬅️ Back to Home", use_container_width=True):
                 with st.spinner("Wait Loading..."):
                    st.session_state.page = "intro"
                    st.rerun()
      
    else:
        # No image uploaded yet
        st.info("👆 Please upload or take a photo of the insect to start identification.")
        
        # Always show back button at bottom when no image
        st.markdown("---")
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            if st.button("⬅️ Back to Home", use_container_width=True, key="back_no_image"):
                 with st.spinner("Wait Loading..."):
                     st.session_state.page = "intro"
                     st.rerun()
# --------------------------------------------------
# Page Routing
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "intro"

if st.session_state.page == "intro":
    intro_page()
elif st.session_state.page == "classification":
    classification_page()
elif st.session_state.page == "about_app":
    about_app_page()
elif st.session_state.page == "features":
    features_page()
elif st.session_state.page == "developers":
    developers_page()

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown("<div class='footer'>© Department of Biotechnology | St. Joseph’s College (Autonomous), Tiruchirappalli</div>", 
            unsafe_allow_html=True)
