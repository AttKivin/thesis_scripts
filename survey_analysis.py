import pandas as pd
import spacy
from collections import Counter
import re

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Load the survey data from the Excel file
data = pd.read_excel('survey_results.xlsx', engine='openpyxl')

# Define column names based on the structure
data.columns = [
    "Timestamp", "PROLIFIC_ID", 
    "Description_1", "Adjectives_1",
    "Description_2", "Adjectives_2",
    "Description_3", "Adjectives_3",
    "Description_4", "Adjectives_4",
    "Description_5", "Adjectives_5",
    "Description_6", "Adjectives_6",
    "Familiarity_AI_Gen", "AI_Art_Real", "Education"
]

# Function to extract candidate adjectives from a string.
def extract_adjectives(adj_string):
    if pd.isna(adj_string):
        return []
    # Normalize: lowercase, trim extra whitespace.
    adj_string = re.sub(r'\s+', ' ', str(adj_string).lower().strip())
    # Split on comma, semicolon, or any whitespace.
    return [item.strip() for item in re.split(r'[,\s;]+', adj_string) if item.strip()]

# Function to tokenize and remove stopwords using spaCy, returning lemmatized tokens.
def tokenize_description(text):
    if pd.isna(text):
        return []
    doc = nlp(str(text).lower())
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

# Updated function to filter candidate adjectives.
# It returns the lemmatized version of the adjective if all tokens are adjectives.
def filter_adjectives(candidates):
    valid = []
    for candidate in candidates:
        doc = nlp(candidate)
        if all(token.pos_ == "ADJ" for token in doc if token.is_alpha):
            # Build the lemmatized version of the candidate.
            lemma_candidate = " ".join([token.lemma_ for token in doc if token.is_alpha])
            valid.append(lemma_candidate)
    return valid

# Function to extract nouns (common themes) using spaCy, returning lemmatized nouns.
def extract_nouns_and_adjectives(text):
    if pd.isna(text):
        return []
    doc = nlp(str(text).lower())
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'PROPN']]

# Define image column pairs (Description and corresponding Adjective columns).
image_pairs = [(f"Description_{i}", f"Adjectives_{i}") for i in range(1, 7)]

# Initialize results for each image.
image_results = {f"Image_{i+1}": {"all_tokens": [], "adjectives": [], "nouns": []} for i in range(6)}
all_tokens, all_adjectives, all_nouns = [], [], []

# Initialize adjective statistics.
adj_stats = {f"Adjectives_{i}": {"cells": 0, "valid_cells": 0} for i in range(1, 7)}

# Overall adjective stats (only for images included in overall counts, i.e. excluding Image 2)
overall_adj_cells = 0
overall_adj_valid_cells = 0

# Process each survey response.
for index, row in data.iterrows():
    for i, (desc_col, adj_col) in enumerate(image_pairs):
        # Process description text.
        tokens = tokenize_description(row[desc_col])
        image_results[f"Image_{i+1}"]["all_tokens"].extend(tokens)
        
        # Extract nouns (common themes) from description.
        nouns = extract_nouns(row[desc_col])
        image_results[f"Image_{i+1}"]["nouns"].extend(nouns)
        
        # Process adjectives provided by the participant.
        if not pd.isna(row[adj_col]):
            # Count this non-empty cell.
            adj_stats[adj_col]["cells"] += 1
            
            # Extract candidate adjectives.
            candidate_adjs = extract_adjectives(row[adj_col])
            valid_adjs = filter_adjectives(candidate_adjs)
            
            # Count as valid if at least one valid adjective is produced.
            if valid_adjs:
                adj_stats[adj_col]["valid_cells"] += 1
            
            image_results[f"Image_{i+1}"]["adjectives"].extend(valid_adjs)
            
            # For overall adjective stats (excluding Image 2, i.e. i != 1)
            if i != 1:
                overall_adj_cells += 1
                if valid_adjs:
                    overall_adj_valid_cells += 1
                all_adjectives.extend(valid_adjs)
        
        # For overall tokens and nouns, exclude Image 2.
        if i != 1:
            all_tokens.extend(tokens)
            all_nouns.extend(nouns)

# Frequency counting using Counter.
image_frequencies = {
    image: {
        "description_word_freq": Counter(results["all_tokens"]),
        "adjective_freq": Counter(results["adjectives"]),
        "noun_freq": Counter(results["nouns"])
    }
    for image, results in image_results.items()
}

overall_description_word_freq = Counter(all_tokens)
overall_adjective_freq = Counter(all_adjectives)
overall_noun_freq = Counter(all_nouns)

# Print adjective statistics per column.
print("\nAdjective Column Statistics (cells processed and valid cells):")
for col in sorted(adj_stats.keys()):
    stats = adj_stats[col]
    print(f"{col}:")
    print(f"  Cells processed: {stats['cells']}")
    print(f"  Valid cells: {stats['valid_cells']}")

# Print overall adjective statistics (from included images).
print("\nOverall Adjective Statistics (from included images):")
print(f"  Cells processed: {overall_adj_cells}")
print(f"  Valid cells: {overall_adj_valid_cells}")

# Data export for CSV implemented with ChatGPT o3-mini-high
# Prepare data for CSV export.
max_rows = max(
    max(len(freq["description_word_freq"]), len(freq["adjective_freq"]), len(freq["noun_freq"]))
    for freq in image_frequencies.values()
)
max_rows = max(max_rows, len(overall_description_word_freq), len(overall_adjective_freq), len(overall_noun_freq))

csv_data = {}
for image in image_frequencies:
    free = image_frequencies[image]["description_word_freq"].most_common(max_rows)
    adj = image_frequencies[image]["adjective_freq"].most_common(max_rows)
    noun = image_frequencies[image]["noun_freq"].most_common(max_rows)
    csv_data.update({
        f"{image}_Free_Word": [w for w, _ in free] + [""] * (max_rows - len(free)),
        f"{image}_Free_Freq": [f for _, f in free] + [""] * (max_rows - len(free)),
        f"{image}_Adj_Word": [w for w, _ in adj] + [""] * (max_rows - len(adj)),
        f"{image}_Adj_Freq": [f for _, f in adj] + [""] * (max_rows - len(adj)),
        f"{image}_Noun_Word": [w for w, _ in noun] + [""] * (max_rows - len(noun)),
        f"{image}_Noun_Freq": [f for _, f in noun] + [""] * (max_rows - len(noun)),
    })

csv_data.update({
    "Overall_Free_Word": [w for w, _ in overall_description_word_freq.most_common(max_rows)] + [""] * (max_rows - len(overall_description_word_freq)),
    "Overall_Free_Freq": [f for _, f in overall_description_word_freq.most_common(max_rows)] + [""] * (max_rows - len(overall_description_word_freq)),
    "Overall_Adj_Word": [w for w, _ in overall_adjective_freq.most_common(max_rows)] + [""] * (max_rows - len(overall_adjective_freq)),
    "Overall_Adj_Freq": [f for _, f in overall_adjective_freq.most_common(max_rows)] + [""] * (max_rows - len(overall_adjective_freq)),
    "Overall_Noun_Word": [w for w, _ in overall_noun_freq.most_common(max_rows)] + [""] * (max_rows - len(overall_noun_freq)),
    "Overall_Noun_Freq": [f for _, f in overall_noun_freq.most_common(max_rows)] + [""] * (max_rows - len(overall_noun_freq)),
})

csv_df = pd.DataFrame(csv_data)
csv_df.to_csv("survey_analysis_results.csv", index=False)
print("\nResults saved to 'survey_analysis_results.csv'")
