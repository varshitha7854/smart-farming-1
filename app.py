import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(page_title="🌿 Smart Farming AI", layout="wide")


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "plant_disease_model.h5"
    return tf.keras.models.load_model(model_path, compile=False)


try:
    model = load_model()
except Exception as exc:
    st.error(f"Failed to load plant_disease_model.h5: {exc}")
    st.stop()

# Class names
class_names = [
    "Pepper Bacterial Spot",
    "Pepper Healthy",
    "PlantVillage",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Bacterial Spot",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]

# Disease information (English)
disease_info = {
    "Pepper Bacterial Spot": {
        "cause": "Xanthomonas bacteria",
        "pesticide": "Copper Hydroxide",
        "dosage": "2 g/L water",
        "prevention": "Use disease-free seeds and avoid overhead irrigation"
    },
    "Potato Early Blight": {
        "cause": "Alternaria fungus",
        "pesticide": "Mancozeb 75% WP",
        "dosage": "2.5 g/L water",
        "prevention": "Remove infected leaves and rotate crops"
    },
    "Potato Late Blight": {
        "cause": "Phytophthora infestans",
        "pesticide": "Copper Oxychloride",
        "dosage": "3 g/L water",
        "prevention": "Avoid excess moisture"
    },
    "Tomato Bacterial Spot": {
        "cause": "Xanthomonas bacteria",
        "pesticide": "Copper-based fungicide",
        "dosage": "2 g/L water",
        "prevention": "Avoid working with wet plants"
    },
    "Tomato Late Blight": {
        "cause": "Fungal infection",
        "pesticide": "Ridomil Gold",
        "dosage": "2.5 g/L water",
        "prevention": "Avoid water on leaves"
    },
    "Tomato Leaf Mold": {
        "cause": "High humidity fungus",
        "pesticide": "Chlorothalonil",
        "dosage": "2 g/L water",
        "prevention": "Improve ventilation"
    },
    "Tomato Septoria Leaf Spot": {
        "cause": "Septoria lycopersici fungus",
        "pesticide": "Mancozeb or Chlorothalonil",
        "dosage": "2 g/L water",
        "prevention": "Mulch soil and avoid overhead watering"
    },
    "Tomato Spider Mites": {
        "cause": "Tetranychus urticae mites",
        "pesticide": "Abamectin or Neem Oil",
        "dosage": "1 mL/L water",
        "prevention": "Keep plants well-watered; avoid dusty conditions"
    },
    "Tomato Target Spot": {
        "cause": "Corynespora cassiicola fungus",
        "pesticide": "Azoxystrobin",
        "dosage": "1 g/L water",
        "prevention": "Remove plant debris and improve air circulation"
    },
    "Tomato Yellow Leaf Curl Virus": {
        "cause": "Begomovirus via whiteflies",
        "pesticide": "Imidacloprid (for whitefly control)",
        "dosage": "0.5 mL/L water",
        "prevention": "Use reflective mulches and sticky yellow traps"
    },
    "Tomato Mosaic Virus": {
        "cause": "Tobacco mosaic virus",
        "pesticide": "No direct cure — remove infected plants",
        "dosage": "N/A",
        "prevention": "Disinfect tools and avoid tobacco near plants"
    }
}

# Disease information (Telugu)
disease_info_telugu = {
    "Pepper Bacterial Spot": {
        "cause": "క్జాంతోమోనాస్ బ్యాక్టీరియా",
        "pesticide": "కాపర్ హైడ్రాక్సైడ్",
        "dosage": "2 గ్రా/లీ నీరు",
        "prevention": "వ్యాధి రహిత విత్తనాలు వాడండి"
    },
    "Potato Early Blight": {
        "cause": "ఆల్టర్నేరియా శిలీంధ్రం",
        "pesticide": "మాంకోజెబ్ 75% WP",
        "dosage": "2.5 గ్రా/లీ నీరు",
        "prevention": "సోకిన ఆకులు తొలగించి, పంట మార్పిడి చేయండి"
    },
    "Potato Late Blight": {
        "cause": "ఫైటోఫ్తోరా ఇన్ఫెస్టాన్స్",
        "pesticide": "కాపర్ ఆక్సీక్లోరైడ్",
        "dosage": "3 గ్రా/లీ నీరు",
        "prevention": "అధిక తేమను నివారించండి"
    },
    "Tomato Bacterial Spot": {
        "cause": "క్జాంతోమోనాస్ బ్యాక్టీరియా",
        "pesticide": "కాపర్ ఆధారిత శిలీంధ్రనాశని",
        "dosage": "2 గ్రా/లీ నీరు",
        "prevention": "తడి మొక్కలతో పని చేయవద్దు"
    },
    "Tomato Late Blight": {
        "cause": "శిలీంధ్ర సంక్రమణ",
        "pesticide": "రిడోమిల్ గోల్డ్",
        "dosage": "2.5 గ్రా/లీ నీరు",
        "prevention": "ఆకులపై నీరు పడకుండా చూసుకోండి"
    },
    "Tomato Leaf Mold": {
        "cause": "అధిక తేమతో శిలీంధ్రం",
        "pesticide": "క్లోరోథలోనిల్",
        "dosage": "2 గ్రా/లీ నీరు",
        "prevention": "గాలి చొరబాటు మెరుగుపరచండి"
    },
    "Tomato Septoria Leaf Spot": {
        "cause": "సెప్టోరియా శిలీంధ్రం",
        "pesticide": "మాంకోజెబ్ లేదా క్లోరోథలోనిల్",
        "dosage": "2 గ్రా/లీ నీరు",
        "prevention": "మల్చింగ్ చేయండి, తల నీటిపారుదల నివారించండి"
    },
    "Tomato Spider Mites": {
        "cause": "సాలీడు పురుగులు",
        "pesticide": "అబమెక్టిన్ లేదా వేప నూనె",
        "dosage": "1 మి.లీ/లీ నీరు",
        "prevention": "మొక్కలకు సరిగ్గా నీరు పెట్టండి"
    },
    "Tomato Target Spot": {
        "cause": "కోరైనెస్పోరా శిలీంధ్రం",
        "pesticide": "అజాక్సీస్ట్రోబిన్",
        "dosage": "1 గ్రా/లీ నీరు",
        "prevention": "మొక్క వ్యర్థాలు తొలగించండి"
    },
    "Tomato Yellow Leaf Curl Virus": {
        "cause": "తెల్లదోమ ద్వారా వైరస్",
        "pesticide": "ఇమిడాక్లోప్రిడ్",
        "dosage": "0.5 మి.లీ/లీ నీరు",
        "prevention": "పరావర్తన మల్చ్ మరియు పసుపు అట్టలు వాడండి"
    },
    "Tomato Mosaic Virus": {
        "cause": "తంబాకు మొజాయిక్ వైరస్",
        "pesticide": "నేరుగా నయం లేదు — సోకిన మొక్కలు తీసివేయండి",
        "dosage": "N/A",
        "prevention": "పనిముట్లు శుభ్రపరచండి"
    }
}

st.title("🌿 Smart Farming AI")
st.write("Detect plant diseases instantly — upload an image or use your camera.")

# Language selection
language = st.selectbox(
    "Select Language / భాష ఎంచుకోండి",
    ["English", "తెలుగు"]
)

st.markdown("---")

# ── INPUT METHOD ──────────────────────────────────────────────
if language == "English":
    input_label = "Choose Input Method"
    upload_tab = "📁 Upload Image"
    camera_tab = "📷 Camera"
else:
    input_label = "ఇన్పుట్ పద్ధతి ఎంచుకోండి"
    upload_tab = "📁 చిత్రం అప్లోడ్ చేయండి"
    camera_tab = "📷 కెమెరా"

input_method = st.radio(input_label, [upload_tab, camera_tab], horizontal=True)

uploaded_file = None

if input_method == upload_tab:
    if language == "English":
        uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"])
        if uploaded_file is None:
            st.info("📸 Please upload a plant leaf image.")
    else:
        uploaded_file = st.file_uploader("ఆకు చిత్రాన్ని అప్లోడ్ చేయండి", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"])
        if uploaded_file is None:
            st.info("📸 దయచేసి మొక్క ఆకు చిత్రాన్ని అప్లోడ్ చేయండి.")

else:  # Camera
    if language == "English":
        st.info("📷 Point your camera at a plant leaf and click the button below to capture.")
        uploaded_file = st.camera_input("Take a photo of the leaf")
    else:
        st.info("📷 మొక్క ఆకు వైపు కెమెరా పెట్టి, క్రింద ఉన్న బటన్ నొక్కి ఫోటో తీయండి.")
        uploaded_file = st.camera_input("ఆకు ఫోటో తీయండి")

# ── PREDICTION ────────────────────────────────────────────────
if uploaded_file is not None:

    # Open image
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        if language == "English":
            st.error("❌ Invalid image file. Please try again.")
        else:
            st.error("❌ చెల్లని చిత్రం. దయచేసి మళ్ళీ ప్రయత్నించండి.")
        st.stop()

    # ── STEP 1: GREEN DOMINANCE CHECK ──
    # Catches selfies, food, random photos before even running the model
    img_np = np.array(image)
    r = img_np[:, :, 0].mean()
    g = img_np[:, :, 1].mean()
    b = img_np[:, :, 2].mean()
    green_dominance = g - ((r + b) / 2)


    if green_dominance < 5:
        if language == "English":
            st.error("❌ Invalid Leaf Image. Please upload a clear green plant leaf.")
            st.info("💡 Tips: Make sure the leaf fills the frame, use good lighting, avoid dark photos.")
        else:
            st.error("❌ చెల్లని ఆకు చిత్రం. దయచేసి స్పష్టమైన పచ్చని ఆకు ఫోటో అప్లోడ్ చేయండి.")
            st.info("💡 సూచన: ఆకు స్పష్టంగా మరియు మంచి వెలుతురులో కనిపించేలా చూసుకోండి.")
        st.stop()

    # Show image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Captured Image" if language == "English" else "తీసిన చిత్రం", width=280)

    # ── STEP 2: PREPROCESS ──
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ── STEP 3: PREDICT ──
    with st.spinner("Analyzing..." if language == "English" else "విశ్లేషిస్తోంది..."):
        prediction = model.predict(img_array)

    index = np.argmax(prediction)
    disease = class_names[index]
    confidence = float(np.max(prediction)) * 100

    # ── STEP 4: ENTROPY CHECK ──
    # Even if image is green, ensure the model is actually confident
    probs = prediction[0]
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    max_entropy = np.log(len(class_names))
    entropy_ratio = entropy / max_entropy  # 0 = certain, 1 = confused


    if confidence < 70 or entropy_ratio > 0.5:
        if language == "English":
            st.error("❌ Could not identify a plant leaf. Please use a clearer image.")
            st.info("💡 Tips: Hold the leaf steady, ensure good lighting, avoid shadows.")
        else:
            st.error("❌ మొక్క ఆకును గుర్తించలేకపోయాం. దయచేసి స్పష్టమైన చిత్రం వాడండి.")
            st.info("💡 సూచన: ఆకును స్థిరంగా పట్టుకోండి, నీడలు లేకుండా చూసుకోండి.")
        st.stop()

    # ── STEP 5: HEALTH SCORE ──
    if "Healthy" in disease:
        health_score = 100
    else:
        health_score = max(10, 100 - int(confidence))

    # ── STEP 6: SHOW RESULTS ──
    with col2:
        if language == "English":
            st.subheader("🌱 Analysis Report")
            st.write(f"**Detected:** {disease}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            if "Healthy" in disease:
                st.success("✅ Plant is Healthy")
                st.write("No pesticide required.")
                st.write("❤️ Health Score: 100 / 100")
            else:
                info = disease_info.get(disease, {
                    "cause": "Unknown",
                    "pesticide": "Consult an agricultural expert",
                    "dosage": "N/A",
                    "prevention": "Monitor the plant closely"
                })
                st.warning("⚠️ Disease Detected")
                st.write(f"**Cause:** {info['cause']}")
                st.write(f"**Pesticide:** {info['pesticide']}")
                st.write(f"**Dosage:** {info['dosage']}")
                st.write(f"**Prevention:** {info['prevention']}")
                st.write("📅 Spray Schedule: Every 7 Days")
                st.write(f"❤️ Health Score: {health_score} / 100")

        else:
            st.subheader("🌱 విశ్లేషణ నివేదిక")
            st.write(f"**గుర్తించింది:** {disease}")
            st.write(f"**ఖచ్చితత్వం:** {confidence:.2f}%")

            if "Healthy" in disease:
                st.success("✅ మొక్క ఆరోగ్యంగా ఉంది")
                st.write("మందు అవసరం లేదు.")
                st.write("❤️ ఆరోగ్య స్కోర్: 100 / 100")
            else:
                info = disease_info_telugu.get(disease, None)
                if info is None:
                    info_en = disease_info.get(disease, {
                        "cause": "తెలియదు",
                        "pesticide": "వ్యవసాయ నిపుణుడిని సంప్రదించండి",
                        "dosage": "N/A",
                        "prevention": "మొక్కను పరిశీలించండి"
                    })
                    info = {
                        "cause": info_en.get("cause", "తెలియదు"),
                        "pesticide": info_en.get("pesticide", "నిపుణుడిని సంప్రదించండి"),
                        "dosage": info_en.get("dosage", "N/A"),
                        "prevention": info_en.get("prevention", "మొక్కను పరిశీలించండి")
                    }
                st.warning("⚠️ వ్యాధి గుర్తించబడింది")
                st.write(f"**కారణం:** {info['cause']}")
                st.write(f"**మందు:** {info['pesticide']}")
                st.write(f"**మోతాదు:** {info['dosage']}")
                st.write(f"**నివారణ:** {info['prevention']}")
                st.write("📅 స్ప్రే షెడ్యూల్: ప్రతి 7 రోజులకు")
                st.write(f"❤️ ఆరోగ్య స్కోర్: {health_score} / 100")

    # Health score progress bar
    st.markdown("### " + ("Plant Health" if language == "English" else "మొక్క ఆరోగ్యం"))
    st.progress(health_score / 100)

st.markdown("---")
st.markdown("Developed using Deep Learning & Streamlit 🚀")
