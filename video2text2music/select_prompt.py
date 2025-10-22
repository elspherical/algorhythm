import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import string

# --- NOTE ON DEPENDENCIES ---
# This script assumes the following NLTK data packages have been
# downloaded once in your environment using a separate Python session:
#
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# ----------------------------

def get_detail_score(prompt: str) -> float:
    """
    Calculates a 'detail score' for a given prompt by assigning weights to
    different parts of speech (POS tags).

    The score is calculated as the sum of weighted scores for all tokens.
    This rewards prompts that are both long and rich in descriptive language.

    The weights prioritize descriptive elements:
    - Adjectives (JJ, JJR, JJS)
    - Proper Nouns (NNP, NNPS)
    - Verbs (VB, VBD, VBG, VBN, VBP, VBZ)
    - Foreign Words (FW) - often technical terms
    """

    # --- Part-of-Speech Tagging Weight Map ---
    # Higher weights mean the word type contributes more to "detail."
    # Based on the Penn Treebank Tagset (used by NLTK's default tagger).
    pos_weights = defaultdict(lambda: 1.0, {
        # Adjectives (Descriptive words)
        'JJ': 3.0,  # Adjective, e.g., 'big', 'dark'
        'JJR': 3.5, # Adjective, comparative, e.g., 'bigger'
        'JJS': 4.0, # Adjective, superlative, e.g., 'biggest'

        # Proper Nouns (Specific entities/names)
        'NNP': 2.5, # Proper noun, singular, e.g., 'Gemini', 'Paris'
        'NNPS': 2.5, # Proper noun, plural, e.g., 'Americans'

        # Foreign words (Often technical or specific terms)
        'FW': 2.0,  # Foreign word, e.g., 'status quo', 'ad hoc'

        # Nouns (Objects/Concepts)
        'NN': 1.5,  # Noun, singular or mass, e.g., 'table', 'water'
        'NNS': 1.5, # Noun, plural, e.g., 'tables'

        # Verbs (Actions)
        'VB': 1.0,  # Base form, e.g., 'take'
        'VBD': 1.0, # Past tense, e.g., 'took'
        'VBG': 1.0, # Gerund/present participle, e.g., 'taking'
    })

    # 1. Tokenize the prompt
    # Remove punctuation before tokenizing for cleaner word analysis
    cleaned_prompt = prompt.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(cleaned_prompt)

    if not tokens:
        return 0.0

    # 2. Part-of-Speech Tagging
    # If the NLTK data is missing, this is where the LookupError occurs.
    tagged_words = nltk.pos_tag(tokens)

    # 3. Calculate Final Weighted Score (Sum of all token weights)
    total_score = 0.0
    for word, tag in tagged_words:
        # Get weight based on POS tag, defaulting to 1.0 for unlisted tags
        weight = pos_weights[tag]
        total_score += weight

    # We now return the raw total score, rewarding longer, more descriptive prompts.
    return total_score

def find_most_detailed_prompt(results):
    """
    Analyzes an array of prompts to find the one with the highest detail score.

    Args:
        prompts: A list of string prompts to analyze.

    Returns:
        A tuple containing the most detailed prompt (string) and its score (float).
    """
    if not results:
        return ("", 0.0)
    

    max_score = -1.0
    most_detailed_prompt = ""

    print("--- Prompt Analysis ---")
    for re in results:
        score = get_detail_score(re[2])
        print(f"Prompt: '{re[2]}'")
        print(f"Detail Score: {score:.2f}")

        if score > max_score:
            max_score = score
            most_detailed_prompt = re[2]

    print("-----------------------")
    return most_detailed_prompt, max_score

