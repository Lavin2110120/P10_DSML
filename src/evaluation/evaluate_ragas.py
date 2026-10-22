# src/evaluation/evaluate_ragas.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
from datasets import Dataset
import pandas as pd

# 1. Charger les questions
questions_df = pd.read_csv("data/evaluation/questions_metier.csv")

# 2. Interroger le système RAG existant (via votre MistralChat ou API interne)
# Pour chaque question, récupérer : réponse générée + contextes récupérés
results = []
for _, row in questions_df.iterrows():
    # Appeler votre système RAG
    response = votre_systeme_rag.query(row['question'])
    results.append({
        "question": row['question'],
        "answer": response['answer'],
        "contexts": response['contexts'],  # Liste des chunks récupérés
        "ground_truth": row['ground_truth_expected']
    })

# 3. Évaluer avec RAGAS
dataset = Dataset.from_pandas(pd.DataFrame(results))
scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_relevancy]
)

# 4. Sauvegarder les résultats
scores.to_pandas().to_csv("reports/ragas_scores_baseline.csv")
print(scores)