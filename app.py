from flask import Flask, render_template, request
from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from IPython.display import Markdown, display

os.environ["OPENAI_API_KEY"] = "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"

app = Flask(__name__)

# Konstruiere den Index
max_input_size = 4096
num_outputs = 300
max_chunk_overlap = 20
chunk_size_limit = 600
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
documents = SimpleDirectoryReader(directory_path).load_data()
index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
index.save_to_disk('index.json')

@app.route("/")
def home():
    # Rendern der Hauptseite
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    # Verwenden Sie den Index, um eine Antwort auf eine Anfrage zu generieren
    query = request.form["message"]
    response = index.query(query, response_mode="compact")
    return response.response

if __name__ == "__main__":
    app.run()
