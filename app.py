from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load a pre-trained zero-shot classification model
interaction_checker = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def check_interaction(drug, virus):
    input_text = f"Check the interaction between the drug {drug} and the virus {virus}."
    candidate_labels = ["no interaction", "minor interaction", "moderate interaction", "severe interaction"]
    result = interaction_checker(input_text, candidate_labels)
    return result['labels'][0], result['scores'][0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    drug = request.form['drug']
    virus = request.form['virus']
    condition = request.form['condition']
    
    interaction, confidence = check_interaction(drug, virus)
    return render_template('result.html', 
                           drug=drug, 
                           virus=virus, 
                           condition=condition, 
                           interaction=interaction, 
                           confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
