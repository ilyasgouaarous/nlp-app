# app.py
import streamlit as st
import html
from datetime import datetime
import pandas as pd

from database import get_connection
from user import Utilisateur
from model_manager import ModelManager

# ✅ NEW: TTS (Text → Speech)
from tts_service_edge import tts_edge, detect_language

# ✅ NEW: File Upload extractors
from io import BytesIO
from docx import Document
from pypdf import PdfReader

# =========================================================
# ✅ FILE UPLOAD HELPERS
# =========================================================
MAX_CHARS = 30000  # safety limit to avoid huge prompts

def extract_text_from_txt(uploaded_file) -> str:
    data = uploaded_file.read()
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")

def extract_text_from_docx(uploaded_file) -> str:
    doc = Document(BytesIO(uploaded_file.read()))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(parts)

def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(BytesIO(uploaded_file.read()))
    pages_text = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            pages_text.append(t)
    return "\n\n".join(pages_text)

def extract_text_any(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    if name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    raise ValueError("Unsupported file type. Please upload PDF / DOCX / TXT.")


# =========================================================
# ✅ PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Text Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# ✅ THEME (LIGHT / DARK) via Session State
# =========================================================
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"


def apply_theme(theme: str):
    """Inject professional CSS with Light/Dark variables."""
    if theme == "Dark":
        vars_css = """
        :root{
            --bg0:#0b1220;
            --bg1:#0f172a;
            --card:#111c2f;
            --card2:#0c1629;
            --text:#e5e7eb;
            --muted:#9ca3af;
            --border:rgba(255,255,255,.10);
            --shadow:rgba(0,0,0,.45);

            --primary:#6366f1;
            --primary2:#4f46e5;
            --secondary:#10b981;
            --danger:#ef4444;
            --warning:#f59e0b;
        }
        """
    else:
        vars_css = """
        :root{
            --bg0:#f6f7fb;
            --bg1:#ffffff;
            --card:#ffffff;
            --card2:#fbfdff;
            --text:#0f172a;
            --muted:#64748b;
            --border:rgba(15,23,42,.10);
            --shadow:rgba(15,23,42,.10);

            --primary:#4f46e5;
            --primary2:#4338ca;
            --secondary:#10b981;
            --danger:#ef4444;
            --warning:#f59e0b;
        }
        """

    st.markdown(
        f"""
        <style>
        {vars_css}

        /* ---------- App background ---------- */
        .stApp {{
            background: radial-gradient(1200px 600px at 10% 0%, rgba(99,102,241,.18), transparent 60%),
                        radial-gradient(900px 500px at 90% 10%, rgba(16,185,129,.12), transparent 55%),
                        linear-gradient(180deg, var(--bg0), var(--bg0));
            color: var(--text);
        }}

        /* ---------- Global typography ---------- */
        html, body, [class*="css"] {{
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
        }}

        /* Remove default top padding feel */
        .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
        }}

        /* ---------- Header ---------- */
        .main-header {{
            text-align:center;
            padding: 1.75rem 1.25rem;
            background: linear-gradient(90deg, var(--primary), var(--primary2));
            border-radius: 18px;
            color: white;
            box-shadow: 0 14px 40px var(--shadow);
            border: 1px solid rgba(255,255,255,.12);
            margin-bottom: 1.25rem;
        }}
        .main-header h1 {{
            font-size: 2.15rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.02em;
        }}
        .main-header p {{
            margin: .45rem auto 0 auto;
            max-width: 720px;
            font-size: 1.02rem;
            color: rgba(255,255,255,.85);
        }}

        /* ---------- Card ---------- */
        .card {{
            background: linear-gradient(180deg, var(--card), var(--card2));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.35rem 1.35rem;
            box-shadow: 0 14px 30px var(--shadow);
        }}
        .card-title {{
            margin: 0 0 .75rem 0;
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--text);
            letter-spacing: -0.01em;
        }}
        .card-subtitle {{
            margin: -0.25rem 0 1rem 0;
            font-size: .95rem;
            color: var(--muted);
        }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(79,70,229,.22), rgba(16,185,129,.10)),
                        linear-gradient(180deg, var(--bg1), var(--bg1));
            border-right: 1px solid var(--border);
        }}
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p {{
            color: var(--text) !important;
        }}

        .sidebar-brand {{
            padding: 1.25rem 1rem 1rem 1rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1rem;
        }}
        .sidebar-brand h2 {{
            margin: 0;
            font-size: 1.25rem;
            font-weight: 900;
            letter-spacing: -0.02em;
        }}
        .sidebar-brand small {{
            color: var(--muted);
        }}

        /* ---------- Inputs ---------- */
        .stTextInput input, .stTextArea textarea {{
            border-radius: 14px !important;
            border: 1px solid var(--border) !important;
            background: rgba(255,255,255,.02) !important;
            color: var(--text) !important;
        }}
        .stTextInput input:focus, .stTextArea textarea:focus {{
            border-color: rgba(99,102,241,.8) !important;
            box-shadow: 0 0 0 4px rgba(99,102,241,.18) !important;
        }}

        /* ---------- Buttons ---------- */
        .stButton > button {{
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,.10) !important;
            background: linear-gradient(90deg, var(--primary), var(--primary2)) !important;
            color: #fff !important;
            padding: .70rem 1.05rem !important;
            font-weight: 750 !important;
            box-shadow: 0 12px 22px var(--shadow) !important;
            transition: transform .15s ease, filter .15s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            filter: brightness(1.06);
        }}

        /* Download button */
        .stDownloadButton > button {{
            background: linear-gradient(90deg, var(--secondary), #059669) !important;
        }}

        /* ---------- Alerts ---------- */
        div[data-testid="stAlert"] {{
            border-radius: 14px !important;
            border: 1px solid var(--border) !important;
        }}

        /* ---------- Badges ---------- */
        .badge {{
            display:inline-flex;
            align-items:center;
            gap:8px;
            padding: .25rem .7rem;
            border-radius: 999px;
            font-size: .78rem;
            font-weight: 850;
            letter-spacing: .06em;
            text-transform: uppercase;
            border: 1px solid rgba(255,255,255,.12);
        }}
        .badge-gen {{
            background: linear-gradient(90deg, rgba(139,92,246,.95), rgba(79,70,229,.95));
            color: white;
        }}
        .badge-sum {{
            background: linear-gradient(90deg, rgba(16,185,129,.95), rgba(5,150,105,.95));
            color: white;
        }}

        /* ---------- Result container ---------- */
        .result {{
            background: linear-gradient(180deg, rgba(2,6,23,.65), rgba(15,23,42,.85));
            border: 1px solid rgba(255,255,255,.10);
            border-left: 4px solid var(--secondary);
            border-radius: 16px;
            padding: 1.1rem;
            color: #e5e7eb;
            line-height: 1.65;
            max-height: 420px;
            overflow: auto;
        }}
        .result::-webkit-scrollbar {{ width: 10px; }}
        .result::-webkit-scrollbar-track {{ background: rgba(255,255,255,.06); border-radius: 10px; }}
        .result::-webkit-scrollbar-thumb {{ background: rgba(16,185,129,.75); border-radius: 10px; }}

        /* ---------- Divider look ---------- */
        hr {{
            border: none;
            border-top: 1px solid var(--border);
            margin: 1.1rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


apply_theme(st.session_state["theme"])

# =========================================================
# ✅ SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2>🚀 AI Text Studio</h2>
            <small>Professional AI Text Processing</small>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Theme selector
    theme = st.selectbox(
        "🎨 Theme",
        ["Light", "Dark"],
        index=0 if st.session_state["theme"] == "Light" else 1
    )
    if theme != st.session_state["theme"]:
        st.session_state["theme"] = theme
        st.rerun()

    # User info + logout
    if "user" in st.session_state:
        st.markdown(
            f"""
            <div class="card" style="padding: 1rem;">
                <div style="display:flex; gap:12px; align-items:center;">
                    <div style="
                        width: 44px; height: 44px; border-radius: 14px;
                        background: linear-gradient(135deg, rgba(99,102,241,.95), rgba(16,185,129,.75));
                        display:flex; align-items:center; justify-content:center;
                        font-weight: 900; color: white;
                    ">
                        {st.session_state["user"].username[0].upper()}
                    </div>
                    <div style="line-height:1.25;">
                        <div style="font-weight:900;">{st.session_state["user"].username}</div>
                        <div style="color: var(--muted); font-size:.9rem;">{st.session_state["user"].email}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.session_state["page"] = "Login"
            st.rerun()

    st.markdown("### 📌 Navigation")

    menu_items = [
        {"name": "Login", "icon": "🔐"},
        {"name": "Register", "icon": "📝"},
        {"name": "Generator", "icon": "✨"},
        {"name": "Upload", "icon": "📎"},        # ✅ NEW PAGE
        {"name": "Assistant", "icon": "🤖"},
        {"name": "History", "icon": "📜"},
        {"name": "Dashboard", "icon": "📈"},
    ]

    if "page" not in st.session_state:
        st.session_state["page"] = "Login"

    for item in menu_items:
        label = f"{item['icon']} {item['name']}"
        if st.button(label, key=f"nav_{item['name']}", use_container_width=True):
            st.session_state["page"] = item["name"]
            st.rerun()


# =========================================================
# ✅ MAIN CONTENT
# =========================================================

# --------------------- LOGIN ---------------------
if st.session_state["page"] == "Login":
    st.markdown(
        """
        <div class="main-header">
            <h1>Welcome Back 👋</h1>
            <p>Sign in to access AI-powered text processing tools</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([1, 1.25, 1])
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔐 Sign In</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Enter your credentials to continue.</div>', unsafe_allow_html=True)

        email = st.text_input("Email Address", placeholder="your.email@example.com", key="login_email")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")

        if st.button("Sign In", use_container_width=True, key="login_btn"):
            if email and password:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, username, email FROM utilisateurs WHERE email=%s AND password=%s",
                    (email, password)
                )
                user = cursor.fetchone()
                cursor.close()
                conn.close()

                if user:
                    st.session_state["user"] = Utilisateur(user[0], user[1], user[2])
                    st.session_state["page"] = "Generator"
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password")
            else:
                st.warning("⚠️ Please fill in all fields")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("<div style='color:var(--muted);'>Don't have an account?</div>", unsafe_allow_html=True)
        if st.button("Create Account →", use_container_width=True, key="goto_register"):
            st.session_state["page"] = "Register"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# --------------------- REGISTER ---------------------
elif st.session_state["page"] == "Register":
    st.markdown(
        """
        <div class="main-header">
            <h1>Create Account 🚀</h1>
            <p>Join AI Text Studio and unlock powerful features</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([1, 1.25, 1])
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📝 Register</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Create your account in seconds.</div>', unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Choose a username", key="reg_username")
        email = st.text_input("Email Address", placeholder="your.email@example.com", key="reg_email")
        password = st.text_input("Password", type="password", placeholder="Create a strong password", key="reg_password")

        if st.button("Create Account", use_container_width=True, key="reg_btn"):
            if username and email and password:
                conn = get_connection()
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "INSERT INTO utilisateurs(username, email, password) VALUES(%s, %s, %s)",
                        (username, email, password)
                    )
                    conn.commit()
                    user_id = cursor.lastrowid
                    cursor.close()
                    conn.close()

                    st.session_state["user"] = Utilisateur(user_id, username, email)
                    st.session_state["page"] = "Generator"
                    st.success("🎉 Account created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please fill in all fields")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("<div style='color:var(--muted);'>Already have an account?</div>", unsafe_allow_html=True)
        if st.button("Sign In →", use_container_width=True, key="goto_login"):
            st.session_state["page"] = "Login"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# --------------------- GENERATOR ---------------------
elif st.session_state["page"] == "Generator":
    if "user" not in st.session_state:
        st.warning("⚠️ Please login first to access the generator.")
        st.stop()

    st.markdown(
        """
        <div class="main-header">
            <h1>AI Text Generator & Summarizer ✨</h1>
            <p>Generate or summarize text with a clean professional interface</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "model_manager" not in st.session_state:
        st.session_state["model_manager"] = ModelManager()
    model = st.session_state["model_manager"]

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">✍️ Input</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Paste your text and choose the operation.</div>', unsafe_allow_html=True)

        text_input = st.text_area(
            "Enter your text",
            height=260,
            placeholder="Type or paste your text here...",
            key="gen_text_input"
        )

        st.markdown("<hr/>", unsafe_allow_html=True)

        op_col1, op_col2 = st.columns([1, 1])
        with op_col1:
            operation = st.radio(
                "Operation",
                ["Generation", "Summary"],
                horizontal=True,
                key="gen_operation"
            )
        with op_col2:
            if operation == "Generation":
                st.info("Creates new content based on your input.")
            else:
                st.info("Condenses your text while keeping key points.")

        if st.button("🚀 Process Text", use_container_width=True, key="process_btn"):
            if text_input.strip():
                with st.spinner("🤖 Processing..."):
                    if operation == "Generation":
                        result = model.generate_text(text_input)
                    else:
                        result = model.summarize_text(text_input)

                    st.session_state["last_operation"] = operation
                    st.session_state["last_result"] = html.escape(result)
                    st.session_state["last_result_raw"] = result
                    st.session_state["last_result_time"] = datetime.now().strftime("%H:%M:%S")

                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO historique_textes(user_id, texte, operation) VALUES (%s, %s, %s)",
                        (st.session_state["user"].id, result, operation.lower())
                    )
                    conn.commit()
                    cursor.close()
                    conn.close()

                    st.success("✅ Saved to history!")
            else:
                st.warning("⚠️ Please enter some text to process.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Result</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Your processed output appears here.</div>', unsafe_allow_html=True)

        if "last_result" in st.session_state:
            op = st.session_state.get("last_operation", "Generation")
            badge_class = "badge-gen" if op == "Generation" else "badge-sum"
            processed_at = st.session_state.get("last_result_time", datetime.now().strftime("%H:%M:%S"))

            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:.75rem;">
                    <div class="badge {badge_class}">{op}</div>
                    <div style="color:var(--muted); font-size:.9rem;">Processed at {processed_at}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="result">
                    {st.session_state["last_result"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.download_button(
                label="📥 Download (.txt)",
                data=st.session_state["last_result_raw"],
                file_name=f"ai_result_{op.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_result"
            )

            # ✅ TTS
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔊 Voice Reader</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subtitle">Listen to the generated result with configurable voice settings.</div>', unsafe_allow_html=True)

            cA, cB = st.columns([0.55, 0.45])

            with cA:
                lang_mode = st.selectbox(
                    "🌍 Language",
                    ["Auto", "fr", "en", "ar", "es", "de", "it", "pt"],
                    index=0,
                    key="tts_lang_mode",
                )
                voice = st.selectbox(
                    "🎙️ Voice",
                    ["Female", "Male"],
                    index=0,
                    key="tts_voice",
                )

            with cB:
                speed = st.slider(
                    "⚡ Speech speed",
                    min_value=0.75,
                    max_value=1.50,
                    value=1.00,
                    step=0.05,
                    key="tts_speed",
                )

            if st.button("▶️ Lire le résultat", use_container_width=True, key="tts_play"):
                text_to_read = st.session_state.get("last_result_raw", "").strip()
                if not text_to_read:
                    st.warning("⚠️ No text to read.")
                else:
                    lang = detect_language(text_to_read) if lang_mode == "Auto" else lang_mode

                    @st.cache_data(show_spinner=False)
                    def _cached_tts(text: str, lang: str, speed: float, voice: str) -> bytes:
                        return tts_edge(text, lang, voice, speed)

                    with st.spinner("🎧 Generating speech..."):
                        audio_mp3 = _cached_tts(text_to_read, lang, speed, voice)

                    st.success(f"✅ Ready ({lang.upper()} • {voice} • x{speed:.2f})")
                    st.audio(audio_mp3, format="audio/mp3")

        else:
            st.markdown(
                """
                <div class="result" style="display:flex;align-items:center;justify-content:center;">
                    <div style="text-align:center; color:#cbd5e1;">
                        <div style="font-size:2.6rem; margin-bottom:.5rem;">🤖</div>
                        <div style="font-weight:900; font-size:1.1rem;">No result yet</div>
                        <div style="opacity:.85; margin-top:.25rem;">Process a text to see output here.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)


# --------------------- UPLOAD (✅ NEW) ---------------------
elif st.session_state["page"] == "Upload":
    if "user" not in st.session_state:
        st.warning("⚠️ Please login first.")
        st.stop()

    st.markdown(
        """
        <div class="main-header">
            <h1>Upload & Process Files 📎</h1>
            <p>Upload a PDF/DOCX/TXT and summarize or generate text from it</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "model_manager" not in st.session_state:
        st.session_state["model_manager"] = ModelManager()
    model = st.session_state["model_manager"]

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📄 Upload file</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Supported formats: PDF, DOCX, TXT</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False
        )

        operation = st.radio(
            "Operation",
            ["Summary", "Generation"],
            horizontal=True,
            key="upload_operation"
        )

        extra_instruction = st.text_input(
            "Optional instruction (tone/style)",
            placeholder="e.g. short summary, bullet points, professional tone...",
            key="upload_instruction"
        )

        preview = st.checkbox("👀 Preview extracted text", value=False, key="upload_preview")

        st.markdown("<hr/>", unsafe_allow_html=True)

        if st.button("🚀 Process File", use_container_width=True, key="upload_process_btn"):
            if not uploaded:
                st.warning("⚠️ Please upload a file first.")
                st.stop()

            try:
                with st.spinner("📥 Extracting text from file..."):
                    extracted = extract_text_any(uploaded).strip()

                if not extracted:
                    st.error("❌ No extractable text found. If it's a scanned PDF, you need OCR.")
                    st.stop()

                if len(extracted) > MAX_CHARS:
                    extracted = extracted[:MAX_CHARS]
                    st.info(f"ℹ️ Large file detected. Processed only the first {MAX_CHARS:,} characters.")

                if preview:
                    with st.expander("📌 Extracted text (preview)", expanded=False):
                        st.text_area("Extracted", extracted, height=220)

                prompt = extracted if not extra_instruction else f"{extra_instruction}\n\n{extracted}"

                with st.spinner("🤖 Running AI model..."):
                    if operation == "Summary":
                        result = model.summarize_text(prompt)
                        db_op = "summary"
                    else:
                        result = model.generate_text(prompt)
                        db_op = "generation"

                st.session_state["last_operation"] = operation
                st.session_state["last_result_raw"] = result
                st.session_state["last_result"] = html.escape(result)
                st.session_state["last_result_time"] = datetime.now().strftime("%H:%M:%S")

                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO historique_textes(user_id, texte, operation) VALUES (%s, %s, %s)",
                    (st.session_state["user"].id, result, db_op)
                )
                conn.commit()
                cursor.close()
                conn.close()

                st.success("✅ Processed & saved to history!")

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Result</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Output appears here.</div>', unsafe_allow_html=True)

        if "last_result" in st.session_state:
            processed_at = st.session_state.get("last_result_time", datetime.now().strftime("%H:%M:%S"))
            st.markdown(
                f"<div style='color:var(--muted); font-size:.9rem; margin-bottom:.6rem;'>Processed at {processed_at}</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<div class='result'>{st.session_state['last_result']}</div>",
                unsafe_allow_html=True
            )

            st.download_button(
                label="📥 Download (.txt)",
                data=st.session_state["last_result_raw"],
                file_name=f"file_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="upload_download_btn"
            )
        else:
            st.markdown(
                """
                <div class="result" style="display:flex;align-items:center;justify-content:center;">
                    <div style="text-align:center; color:#cbd5e1;">
                        <div style="font-size:2.6rem; margin-bottom:.5rem;">📎</div>
                        <div style="font-weight:900; font-size:1.1rem;">No result yet</div>
                        <div style="opacity:.85; margin-top:.25rem;">Upload a file and process it.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)


# --------------------- HISTORY ---------------------
elif st.session_state["page"] == "History":
    if "user" not in st.session_state:
        st.warning("⚠️ Please login first to view your history.")
        st.stop()

    st.markdown(
        """
        <div class="main-header">
            <h1>Your History 📜</h1>
            <p>Review and manage your previous AI text processing results</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, texte, operation, date_creation FROM historique_textes WHERE user_id=%s ORDER BY date_creation DESC",
        (st.session_state["user"].id,)
    )
    history = cursor.fetchall()
    cursor.close()
    conn.close()

    if history:
        st.markdown(
            f"<div style='color:var(--muted); margin-bottom: .75rem;'>Total: <b>{len(history)}</b></div>",
            unsafe_allow_html=True
        )

        for item_id, texte, operation, date_creation in history:
            op = operation.lower()
            if op == "generation":
                badge_class, op_label = "badge-gen", "Generation"
            elif op == "summary":
                badge_class, op_label = "badge-sum", "Summary"
            else:
                badge_class, op_label = "badge-gen", op.capitalize()

            st.markdown('<div class="card" style="margin-bottom: 1rem;">', unsafe_allow_html=True)

            top1, top2 = st.columns([0.82, 0.18])
            with top1:
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
                        <div style="display:flex; align-items:center; gap:10px;">
                            <div class="badge {badge_class}">{op_label}</div>
                            <div style="color:var(--muted); font-size:.92rem;">
                                {date_creation.strftime('%B %d, %Y at %H:%M')}
                            </div>
                            <div style="color:var(--muted); font-size:.85rem;">ID: #{item_id}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with top2:
                if st.button("🗑️ Delete", key=f"delete_{item_id}", use_container_width=True):
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM historique_textes WHERE id=%s", (item_id,))
                    conn.commit()
                    cursor.close()
                    conn.close()
                    st.success(f"✅ Entry #{item_id} deleted!")
                    st.rerun()

            with st.expander(f"View full text ({len(texte)} characters)", expanded=False):
                st.text_area(
                    "Text Content",
                    value=texte,
                    height=180,
                    key=f"text_{item_id}",
                    disabled=True
                )
                st.download_button(
                    label="📥 Download this result",
                    data=texte,
                    file_name=f"history_{item_id}_{op_label.lower()}_{date_creation.strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=f"dl_{item_id}"
                )

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div class="card" style="text-align:center; padding: 2rem;">
                <div style="font-size:3rem;">📭</div>
                <div style="font-weight:900; font-size:1.2rem; margin-top:.25rem;">No History Yet</div>
                <div style="color:var(--muted); max-width:520px; margin:.5rem auto 1.25rem auto;">
                    Your processed text results will appear here. Go to the Generator page to create your first AI output.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("✨ Go to Generator", use_container_width=True, key="go_generator"):
            st.session_state["page"] = "Generator"
            st.rerun()


# --------------------- ASSISTANT ---------------------
elif st.session_state["page"] == "Assistant":
    if "user" not in st.session_state:
        st.warning("⚠️ Please login first.")
        st.stop()

    st.markdown(
        """
        <div class="main-header">
            <h1>AI Assistant 🤖</h1>
            <p>Ask questions and transform the generated text interactively</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "model_manager" not in st.session_state:
        st.session_state["model_manager"] = ModelManager()
    model = st.session_state["model_manager"]

    context_text = st.session_state.get("last_result_raw", "").strip()
    if not context_text:
        st.info("ℹ️ Generate a text (or Upload a file) first, then come back here.")
        st.stop()

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": "Hi! I can help you transform the generated text. Try: “Transform into a professional email”."}
        ]

    with st.expander("📌 Current context (generated text)", expanded=False):
        st.text_area("Context", value=context_text, height=180, disabled=True)

    for m in st.session_state["chat_messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    st.markdown("#### ⚡ Quick actions")
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)
    quick = None
    if qc1.button("✂️ Shorter"):
        quick = "Summarize even shorter (max 3 bullets)."
    if qc2.button("🧒 Explain to a child"):
        quick = "Explain like I'm 8 years old, with a simple example."
    if qc3.button("📧 Pro email"):
        quick = "Transform this into a professional email with subject + greeting + body + closing."
    if qc4.button("🧩 Add examples"):
        quick = "Add 3 concrete examples to illustrate the ideas."
    if qc5.button("🌍 Translate EN"):
        quick = "Translate to English in a natural, professional tone."

    user_prompt = st.chat_input("Ask something about the text...")
    if quick and not user_prompt:
        user_prompt = quick

    if user_prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                reply = model.assistant_reply(context_text=context_text, user_msg=user_prompt)
                st.markdown(reply)

        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})


# --------------------- DASHBOARD ---------------------
elif st.session_state["page"] == "Dashboard":
    if "user" not in st.session_state:
        st.warning("⚠️ Please login first.")
        st.stop()

    st.markdown(
        """
        <div class="main-header">
            <h1>Usage Dashboard 📈</h1>
            <p>Track how many texts you generate per day and per week</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT operation, date_creation
        FROM historique_textes
        WHERE user_id=%s
        ORDER BY date_creation ASC
        """,
        (st.session_state["user"].id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        st.info("No data yet. Generate some texts first 🙂")
        st.stop()

    df = pd.DataFrame(rows, columns=["operation", "date_creation"])
    df["date_creation"] = pd.to_datetime(df["date_creation"])

    c1, c2, c3 = st.columns([0.45, 0.25, 0.30])
    with c1:
        date_min = df["date_creation"].min().date()
        date_max = df["date_creation"].max().date()
        date_range = st.date_input(
            "📅 Date range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )

    with c2:
        group_mode = st.selectbox("Group by", ["Day", "Week"], index=0)

    with c3:
        split_by_operation = st.checkbox("Split by operation (generation/summary)", value=True)

    start_date, end_date = date_range
    mask = (df["date_creation"].dt.date >= start_date) & (df["date_creation"].dt.date <= end_date)
    df = df.loc[mask].copy()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Texts generated</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Counts based on your history table.</div>', unsafe_allow_html=True)

    if group_mode == "Day":
        df["period"] = df["date_creation"].dt.to_period("D").dt.to_timestamp()
    else:
        df["period"] = df["date_creation"].dt.to_period("W").dt.start_time

    if split_by_operation:
        pivot = (
            df.groupby(["period", "operation"])
              .size()
              .reset_index(name="count")
              .pivot(index="period", columns="operation", values="count")
              .fillna(0)
              .sort_index()
        )
        st.line_chart(pivot)
        st.bar_chart(pivot)
    else:
        series = df.groupby("period").size().sort_index()
        st.line_chart(series)
        st.bar_chart(series)

    total = len(df)
    last_7_days = df[df["date_creation"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))].shape[0]

    k1, k2, k3 = st.columns(3)
    k1.metric("Total in range", total)
    k2.metric("Last 7 days", last_7_days)
    k3.metric("Avg per day", round(total / max((end_date - start_date).days + 1, 1), 2))

    st.markdown("</div>", unsafe_allow_html=True)