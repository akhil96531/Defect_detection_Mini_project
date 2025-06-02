import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import io
import re
import gdown  # Import gdown to download the model

# --- UI Styling ---
def local_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Roboto&display=swap');

        /* Background gradient */
        .stApp {
            background: linear-gradient(135deg,
                #1e3c72,
                #6a11cb,
                #000000,
                #f72585,
                #4caf50
            );
            color: #f0f0f0;
            font-family: 'Roboto', sans-serif;
            min-height: 100vh;
            padding-bottom: 40px;
        }

        /* Container max width and centering */
        .stContainer {
            max-width: 850px;
            margin-left: auto;
            margin-right: auto;
            background: rgba(255, 255, 255, 0.1);
            padding: 20px 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        }

        /* Headings font */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Montserrat', sans-serif;
            color: #ffffff;
        }

        /* File uploader style */
        .stFileUploader > div {
            border: 2px dashed #ffffff;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 30px;
            max-width: 850px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Markdown text color */
        .css-1d391kg p, .css-1d391kg span {
            color: #e0e0e0;
        }

        /* Hover effect on images */
        .stImage > img:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 24px rgba(0,0,0,0.6);
            transition: all 0.3s ease-in-out;
        }

        /* Styled download button */
        .stDownloadButton > button {
            background-color: #6a11cb;
            color: white;
            border-radius: 8px;
            padding: 12px 28px;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(106,17,203,0.7);
            transition: background-color 0.3s ease;
            max-width: 850px;
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-bottom: 40px;
        }

        .stDownloadButton > button:hover {
            background-color: #4a0fa9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

local_css()  # Apply styles

# --- Config and Globals ---
CHECKPOINT_PATH = "casting_defect_model_final.pth"  # Local path to save model
MODEL_URL = "https://drive.google.com/uc?export=download&id=1ef8ej8oa2IApM7hdxKDHPIAhSIJkz4si"  # Google Drive model URL

CLASS_NAMES = [
    'blowholes_defect_front', 'cold_shut_defect_front', 'cracks_defect_front', 'def_front',
    'flash_defect_front', 'misrun_defect_front', 'ok_front', 'porosity_defect_front', 'shrinkage_defect_front'
]

DETAILED_DEFECT_INFO = {
    'ok_front': {
        'summary': "Non-defective casting.",
        'cause': "The casting process was successful with no visible defects.",
        'remedy': "No action needed."
    },
    'def_front': {
        'summary': "General defective casting.",
        'cause': "The casting may have multiple underlying issues such as improper pouring, temperature fluctuations, or impurities.",
        'remedy': "Conduct thorough inspection; improve mold design, maintain temperature control, and ensure raw material quality."
    },
    'porosity_defect_front': {
        'summary': "Small holes caused by trapped gas in the casting.",
        'cause': "Gas entrapment during pouring or solidification due to inadequate venting or moisture presence.",
        'remedy': "Improve venting system; preheat mold; degas molten metal; control pouring speed."
    },
    'shrinkage_defect_front': {
        'summary': "Cavities caused by metal contraction during solidification.",
        'cause': "Insufficient molten metal feed during solidification causing shrinkage cavities.",
        'remedy': "Use proper riser design; control cooling rate; increase molten metal volume."
    },
    'misrun_defect_front': {
        'summary': "Incomplete filling of the mold cavity.",
        'cause': "Molten metal solidifies before completely filling the mold due to low pouring temperature or slow pouring.",
        'remedy': "Increase pouring temperature and speed; improve gating system."
    },
    'flash_defect_front': {
        'summary': "Excess metal leakage on the casting surface.",
        'cause': "Poor mold assembly causing gaps or mismatches in mold halves.",
        'remedy': "Ensure proper mold clamping; improve mold design and maintenance."
    },
    'cracks_defect_front': {
        'summary': "Fractures formed due to stress.",
        'cause': "Thermal stress, mechanical shocks, or improper cooling cause cracks.",
        'remedy': "Optimize cooling rates; avoid mechanical shocks; improve mold material toughness."
    },
    'cold_shut_defect_front': {
        'summary': "Incomplete fusion between metal streams.",
        'cause': "Two streams of molten metal meet but do not fuse due to low temperature or turbulence.",
        'remedy': "Increase pouring temperature; reduce turbulence; redesign gating system."
    },
    'blowholes_defect_front': {
        'summary': "Gas bubbles trapped inside casting causing holes.",
        'cause': "Gas generation due to moisture or chemical reactions during casting.",
        'remedy': "Dry mold materials; improve degassing; use vacuum or pressure casting if possible."
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def load_model():
    # If model is not already downloaded, download it
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Downloading model from {MODEL_URL}...")
        gdown.download(MODEL_URL, CHECKPOINT_PATH, quiet=False)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(img: Image.Image):
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred_class = torch.max(probs, 1)
    return CLASS_NAMES[pred_class.item()], conf.item(), probs.cpu().numpy().flatten()

def severity_from_confidence(confidence: float) -> str:
    if confidence < 0.6:
        return "Low Severity"
    elif confidence < 0.8:
        return "Moderate Severity"
    else:
        return "High Severity"

def severity_color(confidence: float) -> str:
    if confidence < 0.6:
        return "#4CAF50"  # Green
    elif confidence < 0.8:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red

def friendly_defect_name(label: str) -> str:
    if label == 'ok_front':
        return "Non Defective"
    name = re.sub(r'_defect_front$', '', label)
    name = name.replace('_', ' ')
    return f"{name.capitalize()} defect found"

def status_icon(defective: bool) -> str:
    return "✅" if not defective else "❌"

def defect_icon(label: str) -> str:
    if label == "ok_front":
        return ""
    return "⚠️"

# Title with magnifying glass emoji
st.markdown("<h1 style='font-family:Montserrat, sans-serif; color:#fff;'>🔍 Casting Defect Detection App</h1>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload one or more casting images",
    accept_multiple_files=True,
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    results = []

    def make_image_name(idx):
        return f"image_{idx:02d}"

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        image = Image.open(uploaded_file).convert("RGB")
        pred_class, conf, _ = predict(image)

        defective = pred_class != 'ok_front'
        casting_status = "Non Defective" if not defective else "Defective"
        defect_display_name = friendly_defect_name(pred_class)

        defect_info = DETAILED_DEFECT_INFO.get(pred_class, {
            'summary': "No information available.",
            'cause': "No information available.",
            'remedy': "No information available."
        })

        severity_label = severity_from_confidence(conf) if defective else "No Defect"
        color = severity_color(conf) if defective else "#4CAF50"  # Green for non-defective

        new_image_name = make_image_name(idx)

        results.append({
            "Image Name": new_image_name,
            "Casting Status": casting_status,
            "Defect": defect_display_name,
            "Confidence": round(conf, 4),
            "Severity": severity_label,
            "Summary": defect_info['summary'],
            "General Cause": defect_info['cause'],
            "Remedy": defect_info['remedy'],
        })

        with st.container():
            st.markdown(f"<div class='stContainer'>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#ffffff; font-size:28px;'>Image: {new_image_name}</h2>", unsafe_allow_html=True)

            st.image(image, width=350)

            st.markdown(f"<p style='font-size:22px; font-weight:bold; color:#ffffff; margin-top:20px;'>Casting Status: {status_icon(defective)} {casting_status}</p>", unsafe_allow_html=True)

            if defective:
                st.markdown(f"<p style='font-size:20px; font-weight:bold; color:#ffcc00;'>Detected Defect: {defect_icon(pred_class)} {defect_display_name}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:20px; font-weight:bold; color:#ff6600;'>Severity (based on confidence): {severity_label}</p>", unsafe_allow_html=True)

            # Always show confidence and progress bar
            st.markdown(f"<p style='font-size:20px; font-weight:bold; color:#ffffff;'>Confidence: {conf:.4f}</p>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style='background:#ddd; border-radius:10px; width:350px;'>
                    <div style='width:{conf*100}%; background:{color}; padding:8px 0; border-radius:10px; text-align:center; color:white; font-weight:bold; font-size:20px; transition: width 0.5s ease-in-out;'>
                        {int(conf*100)}%
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

            with st.expander("Explanation"):
                st.markdown(f"<p style='font-size:22px; font-weight:bold; color:#ffffff;'>Summary:</p> {defect_info['summary']}", unsafe_allow_html=True)
                if defective:
                    st.markdown(f"<p style='font-size:22px; font-weight:bold; color:#ffffff;'>General Cause:</p> {defect_info['cause']}", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:22px; font-weight:bold; color:#ffffff;'>Remedy:</p> {defect_info['remedy']}", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<hr style='border-color:#555;'>", unsafe_allow_html=True)

    df_results = pd.DataFrame(results)
    csv_buffer = io.StringIO()
    df_results.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download predictions as CSV",
        data=csv_buffer.getvalue(),
        file_name="casting_defect_predictions.csv",
        mime="text/csv",
        help="Download all results as CSV"
    )
