# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# class ModelManager:
#     def __init__(self, model_name="gpt2"):
#         self.model_name = model_name
#         self.tokenizer = None
#         self.model = None

#     def load_model(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

#     def generate_text(self, prompt, max_length=100):
#         if not self.model:
#             self.load_model()
#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         outputs = self.model.generate(**inputs, 
# =max_length)
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

#     def summarize_text(self, text, max_length=100):
#         summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#         return summarizer(text, max_length=max_length)[0]["summary_text"]









# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import torch


# class ModelManager:
#     def __init__(self):
#         # =========================
#         # Generator (BloomZ)
#         # =========================
#         self.gpt_model_name = "bigscience/bloomz-560m"
#         self.gpt_tokenizer = None
#         self.gpt_model = None

#         # =========================
#         # Summarizer (BART)
#         # =========================
#         self.summarizer = None

#         # =========================
#         # Assistant (TinyLlama Chat)
#         # =========================
#         self.assistant_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#         self.assistant_tokenizer = None
#         self.assistant_model = None

#         torch.set_num_threads(4)

#     # =========================================================
#     # 🤖 LOAD ASSISTANT MODEL
#     # =========================================================
#     def load_assistant(self):
#         if self.assistant_model is None:
#             self.assistant_tokenizer = AutoTokenizer.from_pretrained(self.assistant_model_name)
#             self.assistant_model = AutoModelForCausalLM.from_pretrained(
#                 self.assistant_model_name,
#                 torch_dtype=torch.float32,
#                 low_cpu_mem_usage=True
#             )
#             self.assistant_model.eval()

#     # =========================================================
#     # 🤖 ASSISTANT REPLY
#     # =========================================================
#     def assistant_reply(self, context_text: str, user_msg: str, output_lang="fr") -> str:
#         self.load_assistant()

#         ctx = (context_text or "").strip()
#         req = (user_msg or "").strip()

#         if not ctx:
#             return "⚠️ No context text found. Generate a text first."

#         lang = "French" if output_lang == "fr" else "English"

#         prompt = f"""
# <|system|>
# You are a professional AI assistant like ChatGPT.
# You must answer in {lang} only.
# You must follow the user request precisely.

# Rules:
# - If user asks for examples, generate 3 examples.
# - If user asks for email, write a professional email (Subject, Greeting, Body, Closing).
# - If user asks for summary, return bullet points.
# - If user asks to explain to a child, simplify with one example.
# - If user asks translation, return only the translation.

# Context:
# {ctx}
# </system>

# <|user|>
# {req}
# </user>

# <|assistant|>
# """.strip()

#         inputs = self.assistant_tokenizer(prompt, return_tensors="pt")

#         with torch.inference_mode():
#             output = self.assistant_model.generate(
#                 **inputs,
#                 max_new_tokens=300,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_p=0.9,
#                 repetition_penalty=1.2,
#                 pad_token_id=self.assistant_tokenizer.eos_token_id
#             )

#         result = self.assistant_tokenizer.decode(output[0], skip_special_tokens=True)

#         # Remove prompt from output
#         if "<|assistant|>" in result:
#             result = result.split("<|assistant|>")[-1]

#         return result.strip()

#     # =========================================================
#     # ✍️ GENERATOR (BloomZ)
#     # =========================================================
#     def load_gpt(self):
#         if self.gpt_model is None:
#             self.gpt_tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_name)
#             self.gpt_model = AutoModelForCausalLM.from_pretrained(
#                 self.gpt_model_name,
#                 low_cpu_mem_usage=True
#             )
#             self.gpt_model.eval()

#     def generate_text(self, prompt, language="fr", max_new_tokens=300):
#         self.load_gpt()

#         formatted_prompt = f"{prompt}\n\nRéponse détaillée :" if language == "fr" else f"{prompt}\n\nDetailed answer:"
#         inputs = self.gpt_tokenizer(formatted_prompt, return_tensors="pt")

#         with torch.inference_mode():
#             outputs = self.gpt_model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 min_new_tokens=150,
#                 do_sample=True,
#                 temperature=0.65,
#                 repetition_penalty=1.4,
#                 no_repeat_ngram_size=4,
#                 pad_token_id=self.gpt_tokenizer.eos_token_id
#             )

#         result = self.gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return result.replace(formatted_prompt, "").strip()

#     # =========================================================
#     # 📝 SUMMARIZER (BART)
#     # =========================================================
#     def load_summarizer(self):
#         if self.summarizer is None:
#             self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

#     def summarize_text(self, text, max_length=130, min_length=40):
#         self.load_summarizer()
#         summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
#         return summary[0]["summary_text"]




from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class ModelManager:
    def __init__(self):
        self.device = "cpu"
        torch.set_num_threads(4)

        # =========================================================
        # ✍️ GENERATOR + 🤖 ASSISTANT
        # Modèle plus fiable que TinyLlama pour suivre des consignes
        # =========================================================
        self.instruct_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.instruct_tokenizer = None
        self.instruct_model = None

        # =========================================================
        # 📝 SUMMARIZER
        # =========================================================
        self.summarizer_model_name = "facebook/bart-large-cnn"
        self.summarizer = None

    # =========================================================
    # 🔧 LOAD INSTRUCT MODEL
    # =========================================================
    def load_instruct_model(self):
        if self.instruct_model is None:
            try:
                self.instruct_tokenizer = AutoTokenizer.from_pretrained(
                    self.instruct_model_name
                )
                self.instruct_model = AutoModelForCausalLM.from_pretrained(
                    self.instruct_model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.instruct_model.eval()
            except Exception as e:
                raise RuntimeError(f"Erreur chargement modèle instruct: {e}")

    # =========================================================
    # 🔧 LOAD SUMMARIZER
    # =========================================================
    def load_summarizer(self):
        if self.summarizer is None:
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model=self.summarizer_model_name
                )
            except Exception as e:
                raise RuntimeError(f"Erreur chargement modèle de résumé: {e}")

    # =========================================================
    # ✂️ TEXT CHUNKING FOR LONG SUMMARIES
    # =========================================================
    def _chunk_text(self, text: str, chunk_size: int = 700):
        words = text.split()
        if not words:
            return []

        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks

    # =========================================================
    # 🧠 SAFE GENERATION CORE
    # =========================================================
    def _generate_response(
        self,
        prompt: str,
        max_input_tokens: int = 1024,
        max_new_tokens: int = 220,
        temperature: float = 0.3,
        do_sample: bool = False
    ) -> str:
        self.load_instruct_model()

        try:
            inputs = self.instruct_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens
            )

            with torch.inference_mode():
                outputs = self.instruct_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=0.9 if do_sample else None,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.instruct_tokenizer.eos_token_id
                )

            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            result = self.instruct_tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

            return result if result else "⚠️ Aucun résultat généré."
        except Exception as e:
            return f"❌ Erreur génération: {str(e)}"

    # =========================================================
    # ✍️ TEXT GENERATION
    # =========================================================
    def generate_text(self, prompt: str, language: str = "fr", max_new_tokens: int = 220) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return "⚠️ Aucun texte fourni."

        if language == "fr":
            system_msg = (
                "Tu es un assistant professionnel de rédaction. "
                "Rédige une réponse claire, structurée, naturelle et professionnelle en français. "
                "N'invente pas de faits inutiles."
            )
            user_msg = f"Voici la demande utilisateur :\n{prompt}\n\nRédige une réponse complète et professionnelle."
        else:
            system_msg = (
                "You are a professional writing assistant. "
                "Write a clear, structured, natural, and professional response in English. "
                "Do not invent unnecessary facts."
            )
            user_msg = f"Here is the user request:\n{prompt}\n\nWrite a complete and professional response."

        full_prompt = f"""System:
{system_msg}

User:
{user_msg}

Assistant:
""".strip()

        return self._generate_response(
            prompt=full_prompt,
            max_input_tokens=1024,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            do_sample=False
        )

    # =========================================================
    # 📝 SUMMARIZATION
    # =========================================================
    def summarize_text(self, text: str, max_length: int = 130, min_length: int = 40) -> str:
        self.load_summarizer()

        text = (text or "").strip()
        if not text:
            return "⚠️ Aucun texte à résumer."

        try:
            chunks = self._chunk_text(text, chunk_size=700)

            if len(chunks) == 1:
                summary = self.summarizer(
                    chunks[0],
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                return summary[0]["summary_text"]

            partial_summaries = []
            for chunk in chunks:
                summary = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                partial_summaries.append(summary[0]["summary_text"])

            combined_summary = " ".join(partial_summaries)

            final_summary = self.summarizer(
                combined_summary,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return final_summary[0]["summary_text"]

        except Exception as e:
            return f"❌ Erreur résumé: {str(e)}"

    # =========================================================
    # 🤖 ASSISTANT REPLY
    # =========================================================
    def assistant_reply(self, context_text: str, user_msg: str, output_lang: str = "fr") -> str:
        ctx = (context_text or "").strip()
        req = (user_msg or "").strip()

        if not ctx:
            return "⚠️ Aucun texte de contexte trouvé. Génère un texte d'abord."

        if not req:
            return "⚠️ Aucun message utilisateur fourni."

        lang_name = "français" if output_lang == "fr" else "English"

        if output_lang == "fr":
            system_msg = f"""
Tu es un assistant IA professionnel spécialisé en reformulation, transformation et amélioration de texte.
Tu dois répondre uniquement en {lang_name}.

Règles importantes :
- Sois clair, professionnel et bien structuré.
- Respecte précisément la demande de l'utilisateur.
- Base-toi uniquement sur le contexte fourni.
- N'ajoute pas d'informations inventées sauf si l'utilisateur demande explicitement des exemples.
- Si l'utilisateur demande un email, fournis :
  Objet
  Salutation
  Corps
  Formule de politesse
- Si l'utilisateur demande un résumé, réponds sous forme de points.
- Si l'utilisateur demande une traduction, retourne uniquement la traduction.
- Si l'utilisateur demande une explication simple, utilise des mots faciles et un exemple concret.
- Garde un ton professionnel et naturel.
""".strip()
        else:
            system_msg = f"""
You are a professional AI assistant specialized in rewriting, transforming, and improving text.
You must answer only in {lang_name}.

Important rules:
- Be clear, professional, and well structured.
- Follow the user request precisely.
- Base your answer only on the provided context.
- Do not invent information unless the user explicitly asks for examples.
- If the user asks for an email, provide:
  Subject
  Greeting
  Body
  Closing
- If the user asks for a summary, answer in bullet points.
- If the user asks for a translation, return only the translation.
- If the user asks for a simple explanation, use simple words and one concrete example.
- Keep a professional and natural tone.
""".strip()

        full_prompt = f"""System:
{system_msg}

Context:
{ctx}

User:
{req}

Assistant:
""".strip()

        return self._generate_response(
            prompt=full_prompt,
            max_input_tokens=1400,
            max_new_tokens=260,
            temperature=0.2,
            do_sample=False
        )