import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv


# Load API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


import json
import random

def load_examples():
    with open("examples.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_all_examples():
    examples = load_examples()
    
    formatted = ""
    for ex in examples:
        formatted += f"\nAI:\n{ex['ai']}\n\nHuman:\n{ex['human']}\n\n---\n"
    
    return formatted



# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Humanizer", layout="wide")

st.markdown("""
    <h2 style='text-align: center; color: #EB0525;'>
    🤖 AI TO Human Text Converter
    </h2>
    <hr>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    [
        "llama-3.3-70b-versatile",
        #"deepseek-r1-distill-llama-70b",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m",

        "moonshotai/kimi-k2-instruct-0905",

        "openai/gpt-oss-120b",
        "whisper-large-v3-turbo",

        "groq/compound",
        "qwen/qwen3-32b"
    ], index=0
)

# ---------------- INPUT ----------------
st.subheader("📥 Input AI Text")

input_text = st.text_area(
    "Paste AI-generated text here:",
    height=250
)

# ---------------- PROMPT ----------------
def build_prompt(text):
    examples_block = get_all_examples()

    return f"""
You are rewriting AI-generated text to sound like it was written by a real student or young professional.

STRICT RULES:
- Write like a real person talking — not an essay bot
- Use contractions (it's, they're, don't, that's)
- Vary sentence length a lot: mix very short sentences with longer ones
- Occasionally start sentences with "And", "But", or "So"
- Use everyday words — never say "utilize", "facilitate", "leverage", "delve"
- Add small human touches: "basically", "pretty much", "it turns out", "thing is"
- Break one long idea into two short punchy sentences sometimes
- Do NOT use bullet points, headers, or numbered lists
- Do NOT use transition words like "Furthermore", "Moreover", "Additionally", "In conclusion"
- Do NOT summarize at the end
- Output length must stay within 30% of the original

Examples of AI and Human rewrites:
{examples_block}

Now rewrite this text:
{text}

Return ONLY the rewritten text. Nothing else.
"""

#-----------------------------------------------------------------------------------------------------
import re

def post_process(text):
    # Replace overused AI transition words
    replacements = {
    r'\butilize\b': 'use',
    r'\butilization\b': 'use',
    r'\bfacilitate\b': 'help',
    r'\bleverage\b': 'use',
    r'\bdelve\b': 'dig into',
    r'\bsubstantial\b': 'big',
    r'\bsubstantially\b': 'a lot',
    r'\bsignificant\b': 'major',
    r'\bsignificantly\b': 'a lot',
    r'\bdemonstrate\b': 'show',
    r'\bdemonstrated\b': 'shown',
    r'\bpivotal\b': 'key',
    r'\bcrucial\b': 'important',
    r'\bcomprehensive\b': 'full',
    r'\bimplementation\b': 'use',
    r'\bprolifer\w+\b': 'spread',
    r'\byield\b': 'give',
    r'\bmitigate\b': 'reduce',
    r'\badverse\b': 'bad',
    r'\bpredominantly\b': 'mostly',
    r'\bsubsequently\b': 'then',
    r'\bnevertheless\b': 'still',
    r'\bnotwithstanding\b': 'even so',
    r'\bhence\b': 'so',
    r'\bthus\b': 'so',
    r'\btherefore\b': 'so',
    r'\bmoreover\b': 'also',
    r'\bfurthermore\b': 'also',
    r'\badditionally\b': 'and',
    r'\bin conclusion\b': 'so',
    r'\bin summary\b': 'basically',
    r'\bit is worth noting that\b': 'worth knowing —',
    r'\bit is important to note that?\b': 'keep in mind,',
    r'\bit is evident that\b': 'clearly',
    r'\bthe fact that\b': 'that',
    r'\bin order to\b': 'to',
    r'\ba wide range of\b': 'many',
    r'\ba variety of\b': 'many',
    r'\bnumerous\b': 'many',
    r'\bindividuals\b': 'people',
    r'\bpopulace\b': 'people',
    r'\bcommence\b': 'start',
    r'\binitiate\b': 'start',
    r'\bterminate\b': 'end',
    r'\bprocure\b': 'get',
    r'\benhance\b': 'improve',
    r'\boptimal\b': 'best',
    r'\bprimarily\b': 'mainly',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text
#-----------------------------------------------------------------------------------------------------

# ---------------- LLM ----------------
def get_response(prompt):
    llm = ChatGroq(
        temperature=1.4,  # for Randomness
        groq_api_key=api_key,
        model=model_choice
    )
    return llm.invoke(prompt).content

# ---------------- BUTTON ----------------
if st.button("Humanize Text"):
    if input_text.strip() == "":
        st.warning("Please enter some text")
    else:
        with st.spinner("Making it human-like........."):
            prompt = build_prompt(input_text)
            output = get_response(prompt)
            output = post_process(output)

        st.subheader("📤 Humanized Output")
        st.write(output)


# ---------------- FOOTER ----------------
st.markdown("---")