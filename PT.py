import torch
import gradio as gr
import json
from transformers import pipeline, AutoTokenizer

print("Loading Summarization model: t5-base...")
text_summary = pipeline(
    "summarization",
    model="t5-base",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)
print("Summarization model loaded.")

def summarize_text(input_text):
    output = text_summary(input_text, min_length=30, max_length=120)
    return output[0]['summary_text']

print("Loading Translation model: facebook/nllb-200-distilled-600M...")
text_translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)
print("Translation model loaded.")

with open('language.json', 'r', encoding='utf-8') as file:
    language_data = json.load(file)

def get_FLORES_code_from_language(language_name):
    for entry in language_data:
        if entry['Language'].lower() == language_name.lower():
            return entry['FLORES-200 code']
    return None

def translate_text(text, destination_language_name):
    dest_code = get_FLORES_code_from_language(destination_language_name)

    if not dest_code:
        return f"Error: FLORES-200 code not found for '{destination_language_name}'. Please select a language from the list."

    print(f"Translating from English (eng_Latn) to {destination_language_name} ({dest_code})...")

    translation = text_translator(
        text,
        src_lang="eng_Latn",
        tgt_lang=dest_code
    )
    return translation[0]["translation_text"]

print("Loading Question-Answering model: distilbert-base-uncased-distilled-squad...")
Youtube = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)
print("Question-Answering model loaded.")

def read_file_content(file_obj):
    try:
        with open(file_obj.name, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(file_obj.name, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            return f"An error occurred: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

def get_answer(file, question):
    context = read_file_content(file)
    if "An error occurred" in context:
        return context
    answer = Youtube(question=question, context=context)
    if answer['score'] < 0.2:
        return "I am not confident I can answer this question based on the provided text."
    return answer["answer"]

def landing_page():
    return (
        "<div style='max-width:900px; margin:36px auto 0 auto; background: linear-gradient(135deg, #eef6fb 0%, #e0ecfa 100%); "
        "border-radius: 30px; box-shadow:0 8px 36px #b6c4e033; padding: 60px 0 45px 0; text-align:center;'>"
        "<h1 style='font-size:3.2em; color:#174ea6; font-weight:900; letter-spacing:-1.5px; margin-bottom:10px;'>Unified NLP Toolkit ✨</h1>"
        "<p style='font-size:1.27em; color:#51627a; margin:24px 0 27px 0;'>"
        "Summarize, translate, or ask questions on documents using a modern, friendly interface.<br><br>"
        "Select a tool below to get started."
        "</p>"
        "<div style='display:flex; flex-wrap:wrap; justify-content:center; gap:30px; margin-top:27px;'>"
        "<a href='#summarizer' style='background:linear-gradient(100deg,#2563eb 70%,#4f92fc 100%) !important; "
        "color:white; padding:17px 36px 17px 36px; border-radius:16px; text-decoration:none; font-size:1.16em; font-weight:700; "
        "transition: box-shadow .15s; box-shadow:0 4px 14px #2563eb22; outline:none;'>Text Summarizer</a>"
        "<a href='#translator' style='background:linear-gradient(100deg,#059669 70%,#34d399 110%) !important; "
        "color:white; padding:17px 36px 17px 36px; border-radius:16px; text-decoration:none; font-size:1.16em; font-weight:700; "
        "transition: box-shadow .15s; box-shadow:0 4px 14px #05966922; outline:none;'>Translator</a>"
        "<a href='#qna' style='background:linear-gradient(100deg,#f59e42 70%,#fbbf24 110%) !important; "
        "color:white; padding:17px 36px 17px 36px; border-radius:16px; text-decoration:none; font-size:1.16em; font-weight:700; "
        "transition: box-shadow .15s; box-shadow:0 4px 14px #f59e4222; outline:none;'>Document Q&A</a>"
        "</div>"
        "</div>"
    )

def footer():
    return (
        "<div style='text-align:center; margin-top:54px; margin-bottom:12px; color:#64748b; font-size:1.13em; letter-spacing:0.5px;'>"
        "Made with <span style='color:#e11d48;'>&#10084;&#65039;</span> by tushar"
        "</div>"
    )

with gr.Blocks(title="Unified NLP Toolkit ✨", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.HTML(landing_page())

    with gr.Tab("Text Summarizer", elem_id="summarizer"):
        gr.Markdown("### Text Summarizer")
        gr.Markdown("> Enter your text below and click **Summarize** to generate a concise summary.")
        inp = gr.Textbox(
            label="Input",
            lines=10,
            placeholder="Paste your text here..."
        )
        btn = gr.Button(
            "Summarize",
            elem_id="summarize-btn",
            variant="secondary"
        )
        out = gr.Textbox(label="Summary", lines=4)
        btn.click(fn=summarize_text, inputs=inp, outputs=out)

    with gr.Tab("Multilanguage Translator", elem_id="translator"):
        gr.Markdown("### Multilanguage Translator")
        gr.Markdown("> Enter English text, select an Indian language, and click **Translate**.")
        inp3 = gr.Textbox(
            label="English Text",
            lines=10
        )
        lang_options = [entry['Language'] for entry in language_data]
        lang_dropdown = gr.Dropdown(
            lang_options,
            label="Select Indian Language"
        )
        btn3 = gr.Button(
            "Translate",
            elem_id="translate-btn",
            variant="secondary"
        )
        out3 = gr.Textbox(label="Translated Text", lines=4)
        btn3.click(fn=translate_text, inputs=[inp3, lang_dropdown], outputs=out3)

    with gr.Tab("Document QnA", elem_id="qna"):
        gr.Markdown("### Document Q&A")
        gr.Markdown("> Upload a `.txt` file and ask a question about its content.")
        file_inp = gr.File(label="Upload Text File")
        q_inp = gr.Textbox(label="Your Question", lines=2)
        btn4 = gr.Button(
            "Get Answer",
            elem_id="qna-btn",
            variant="secondary"
        )
        out4 = gr.Textbox(label="Answer", lines=2)
        btn4.click(fn=get_answer, inputs=[file_inp, q_inp], outputs=out4)

    gr.HTML(footer())

demo.launch()
