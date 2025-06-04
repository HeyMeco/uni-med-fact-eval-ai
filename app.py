import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import re
import requests
import os
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv(override=False)  # Don't override existing system environment variables

# Set page config
st.set_page_config(
    page_title="Medical Article Evaluation Visualizer",
    page_icon="🏥",
    layout="wide"
)

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')  # Use os.environ.get instead of os.getenv
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def generate_html_color(text, saturation=0.7, lightness_base=0.8, lightness_keyword=0.9):
    """Generates a somewhat consistent color pair based on text input."""
    hash_object = hashlib.md5(text.encode())
    hex_dig = hash_object.hexdigest()
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)

    # Normalize and adjust lightness
    base_r = int(r + (255 - r) * (1 - saturation) + (255-r) * (1-lightness_base) ) // 2
    base_g = int(g + (255 - g) * (1 - saturation) + (255-g) * (1-lightness_base) ) // 2
    base_b = int(b + (255 - b) * (1 - saturation) + (255-b) * (1-lightness_base) ) // 2
    base_color = f'#{min(255,base_r):02x}{min(255,base_g):02x}{min(255,base_b):02x}'

    key_r = int(r + (255 - r) * (1 - saturation) + (255-r) * (1-lightness_keyword) ) // 2
    key_g = int(g + (255 - g) * (1 - saturation) + (255-g) * (1-lightness_keyword) ) // 2
    key_b = int(b + (255 - b) * (1 - saturation) + (255-b) * (1-lightness_keyword) ) // 2
    keyword_color = f'#{min(255,key_r):02x}{min(255,key_g):02x}{min(255,key_b):02x}'

    return base_color, keyword_color

# Define aspect labels
ASPECT_LABELS = {
    'ob': 'Objective',
    'i': 'Intervention',
    'c': 'Comparator',
    'b': 'Blinding Method',
    'p': 'Population',
    'm': 'Medicines',
    'td': 'Treatment Duration',
    'pe': 'Primary Endpoints',
    'fd': 'Follow-up Duration',
    'o': 'Outcomes',
    'f': 'Findings',
    'rf': 'Reference'
}

# Define colors for different aspects (base_color, keyword_color)
ASPECT_COLORS = {
    "ob": ("#FFADAD", "#FFD6A5"),  # Light Red, Lighter Red/Orange
    "i": ("#A0C4FF", "#BDB2FF"),   # Light Blue, Lighter Blue/Purple
    "c": ("#9BF699", "#FDFFB6"),   # Light Green, Lighter Green/Yellow
    "b": ("#FFC6FF", "#CAFFBF"),   # Light Pink, Lighter Pink/Mint
    "p": ("#FFD6A5", "#FFE5B4"),   # Light Orange, Lighter Orange (peach)
    "m": ("#BDB2FF", "#E0BBE4"),   # Light Purple, Lighter Purple (lilac)
    "td": ("#FDFFB6", "#FFFFE0"),  # Light Yellow, Lighter Yellow (lemon chiffon)
    "pe": ("#CAFFBF", "#E6FFFA"),  # Light Mint, Lighter Mint (alice blueish)
    "fd": ("#FFC6FF", "#FFDDF4"),  # Light Magenta, Lighter Magenta
    "o": ("#A0C4FF", "#CDE1FF"),   # Repeating Light Blue, Lighter
    "f": ("#9BF699", "#D4FAD1"),   # Repeating Light Green, Lighter
    "rf": ("#FFADAD", "#FFCBCB"),  # Repeating Light Red, Lighter
}

# This will be populated as aspects are encountered
USED_ASPECT_COLORS = {}

def get_aspect_colors(aspect_code):
    if aspect_code not in USED_ASPECT_COLORS:
        if aspect_code in ASPECT_COLORS:
            USED_ASPECT_COLORS[aspect_code] = ASPECT_COLORS[aspect_code]
        else:
            print(f"Warning: Aspect '{aspect_code}' not in predefined colors. Generating one.")
            USED_ASPECT_COLORS[aspect_code] = generate_html_color(aspect_code)
    return USED_ASPECT_COLORS[aspect_code]

# Define star rating options
STAR_RATINGS = {
    1: "⭐",
    2: "⭐⭐",
    3: "⭐⭐⭐",
    4: "⭐⭐⭐⭐",
    5: "⭐⭐⭐⭐⭐"
}

# Define the JSON schema for structured output
EVALUATION_SCHEMA = {
    "name": "medical_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "ratings": {
                "type": "object",
                "properties": {
                    "com_eval_summary": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Completeness rating for the summary (1-5)"
                    },
                    "con_eval_summary": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Conciseness rating for the summary (1-5)"
                    },
                    "com_eval_sentences": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Completeness rating for the sentences (1-5)"
                    },
                    "con_eval_sentences": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Conciseness rating for the sentences (1-5)"
                    },
                    "com_eval_kps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Completeness rating for key phrases (1-5)"
                    },
                    "con_eval_kps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Conciseness rating for key phrases (1-5)"
                    }
                },
                "required": [
                    "com_eval_summary",
                    "con_eval_summary",
                    "com_eval_sentences",
                    "con_eval_sentences",
                    "com_eval_kps",
                    "con_eval_kps"
                ],
                "additionalProperties": False
            },
            "key_elements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sentence": {
                            "type": "integer",
                            "description": "The index of the sentence containing the key element"
                        },
                        "index": {
                            "type": "integer",
                            "description": "The index of the word within the sentence"
                        }
                    },
                    "required": ["sentence", "index"],
                    "additionalProperties": False
                },
                "description": "Array of key elements identified in the text"
            },
            "explanation": {
                "type": "object",
                "properties": {
                    "ratings_rationale": {
                        "type": "string",
                        "description": "Detailed explanation of the ratings given"
                    },
                    "key_elements_selection": {
                        "type": "string",
                        "description": "Explanation of how key elements were identified"
                    }
                },
                "required": ["ratings_rationale", "key_elements_selection"],
                "additionalProperties": False
            }
        },
        "required": ["ratings", "key_elements", "explanation"],
        "additionalProperties": False
    }
}

def load_json_data(file_path='35964471.json'):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except Exception as e:
        st.error(f"Error loading JSON file: {e}")
        return None

def create_evaluation_radar_chart(evaluations, title):
    categories = ['Summary', 'Sentences', 'Key Phrases']
    completeness_values = [
        evaluations['com_eval_summary'],
        evaluations['com_eval_sentences'],
        evaluations['com_eval_kps']
    ]
    conciseness_values = [
        evaluations['con_eval_summary'],
        evaluations['con_eval_sentences'],
        evaluations['con_eval_kps']
    ]

    fig = go.Figure()
    
    # Add completeness trace
    fig.add_trace(go.Scatterpolar(
        r=completeness_values,
        theta=categories,
        fill='toself',
        name='Completeness'
    ))
    
    # Add conciseness trace
    fig.add_trace(go.Scatterpolar(
        r=conciseness_values,
        theta=categories,
        fill='toself',
        name='Conciseness'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=True,
        title=title
    )
    
    return fig

def create_rating_section(title, prefix, initial_values):
    st.subheader(title)
    
    # Add explanation of the evaluation criteria
    if prefix == "completeness":
        st.markdown("""
        **Evaluation Criteria:**
        Every piece of information is critical in medical text. The system output must be entirely complete, with no omissions.
        
        **Rating Scale:**
        - 5 Stars: All information has been accurately found, cannot find any more relevant information
        - 4 Stars: Most key information found, small amount of information missing
        - 3 Stars: Some information found, some information missing
        - 2 Stars: Most key information missing
        - 1 Star: All key information missing
        """)
    else:  # conciseness
        st.markdown("""
        **Evaluation Criteria:**
        The system output must be fully accurate, without any errors.
        
        **Rating Scale:**
        - 5 Stars: All content is relevant to this aspect
        - 4 Stars: Most content is relevant to this aspect
        - 3 Stars: Some content is relevant, some irrelevant
        - 2 Stars: Most content is irrelevant to this aspect
        - 1 Star: All content is irrelevant or contains errors
        """)
    
    # Create columns for the three rating categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Summary**")
        if prefix == "completeness":
            st.markdown("*Does the summary include all claims from the abstract on this aspect?*")
        else:
            st.markdown("*Does the summary contain irrelevant/incorrect information?*")
        summary_rating = st.selectbox(
            "Summary Rating",
            options=list(STAR_RATINGS.keys()),
            format_func=lambda x: STAR_RATINGS[x],
            key=f"{prefix}_summary",
            index=initial_values['summary']-1
        )
        
    with col2:
        st.markdown("**Sentences**")
        if prefix == "completeness":
            st.markdown("*Have all sentences related to this aspect been highlighted?*")
        else:
            st.markdown("*Are there highlighted sentences irrelevant to this aspect?*")
        sentences_rating = st.selectbox(
            "Sentences Rating",
            options=list(STAR_RATINGS.keys()),
            format_func=lambda x: STAR_RATINGS[x],
            key=f"{prefix}_sentences",
            index=initial_values['sentences']-1
        )
        
    with col3:
        st.markdown("**Key Phrases**")
        if prefix == "completeness":
            st.markdown("*Have all key phrases related to this aspect been highlighted?*")
        else:
            st.markdown("*Are there highlighted key phrases irrelevant to this aspect?*")
        kps_rating = st.selectbox(
            "Key Phrases Rating",
            options=list(STAR_RATINGS.keys()),
            format_func=lambda x: STAR_RATINGS[x],
            key=f"{prefix}_kps",
            index=initial_values['kps']-1
        )
    
    return {
        'summary': summary_rating,
        'sentences': sentences_rating,
        'kps': kps_rating
    }

def highlight_article_text(article_text, article_tokens, selected_aspect_data):
    """Highlights article text using token-based approach with sentence and keyword colors."""
    if not selected_aspect_data or 'kes' not in selected_aspect_data:
        return ' '.join(article_text)

    num_sentences = len(article_tokens)
    token_styles = [
        [{"sentence_color": None, "keyword_color": None} for _ in range(len(article_tokens[s_idx]))]
        for s_idx in range(num_sentences)
    ]

    aspect = selected_aspect_data.get("aspect")
    if not aspect:
        return ' '.join(article_text)

    aspect_sentence_color, aspect_keyword_color = get_aspect_colors(aspect)
    kes_list = selected_aspect_data.get("kes", [])

    # First, mark all sentences that contain key elements
    involved_sentence_indices = set()
    for ke_item in kes_list:
        s_idx = ke_item.get("sentence")
        if s_idx is not None and 0 <= s_idx < num_sentences:
            involved_sentence_indices.add(s_idx)

    # Apply sentence-level highlighting
    for s_idx in involved_sentence_indices:
        for t_idx in range(len(article_tokens[s_idx])):
            if token_styles[s_idx][t_idx]["sentence_color"] is None:
                token_styles[s_idx][t_idx]["sentence_color"] = aspect_sentence_color

    # Apply keyword-level highlighting
    for ke_item in kes_list:
        s_idx = ke_item.get("sentence")
        t_idx = ke_item.get("index")

        if s_idx is not None and t_idx is not None and \
           0 <= s_idx < num_sentences and 0 <= t_idx < len(article_tokens[s_idx]):
            token_styles[s_idx][t_idx]["keyword_color"] = aspect_keyword_color

    # Generate HTML output
    html_output = []
    for s_idx, sentence_tokens in enumerate(article_tokens):
        styled_sentence_parts = []
        for t_idx, token_text in enumerate(sentence_tokens):
            style_info = token_styles[s_idx][t_idx]
            bg_color = style_info["keyword_color"] or style_info["sentence_color"]
            safe_token_text = token_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            if bg_color:
                text_color = "#000000"  # Black text for better readability
                styled_sentence_parts.append(
                    f'<span style="background-color: {bg_color}; padding: 0.2em 0;">{safe_token_text}</span>'
                )
            else:
                styled_sentence_parts.append(safe_token_text)

        # Reconstruct sentence with proper spacing
        reconstructed_sentence = ""
        for i, part in enumerate(styled_sentence_parts):
            original_token = sentence_tokens[i]
            if i > 0 and not (original_token in ['.', ',', ';', ':', ')', ']', "'s", "n't", "'"] or \
               (len(sentence_tokens[i-1]) > 0 and sentence_tokens[i-1] in ['('])):
                reconstructed_sentence += " "
            reconstructed_sentence += part

        html_output.append(reconstructed_sentence)

    return ' '.join(html_output)

def evaluate_with_openrouter(text, aspect, aspect_data=None, article_tokens=None):
    """
    Use OpenRouter API to evaluate the text for a specific aspect using structured outputs.
    """
    # Create a placeholder for the debug information
    debug_container = st.expander("🔍 Debug Information", expanded=True)
    
    with debug_container:
        st.write("### API Configuration")
        st.json({
            "API_URL": OPENROUTER_URL,
            "Model": st.session_state.selected_model,
            "Has_API_Key": bool(OPENROUTER_API_KEY)
        })

    if not OPENROUTER_API_KEY:
        st.error("OpenRouter API key not found. Please set it either in the .env file or as a system environment variable named 'OPENROUTER_API_KEY'.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:8501",
    }

    # Create a system message to enforce JSON output
    system_message = """You are a medical text evaluation assistant. You must evaluate medical text according to these specific criteria:

COMPLETENESS EVALUATION:
Every piece of information is critical in medical text. The system output must be entirely complete, with no omissions.

Rating Scale for Completeness:
5 Stars: All information has been accurately found, cannot find any more relevant information
4 Stars: Most key information found, small amount of information missing
3 Stars: Some information found, some information missing
2 Stars: Most key information missing
1 Star: All key information missing

Evaluate completeness by answering:
[Summary] Does the summary include all claims from the abstract on this aspect?
[Sentences] Have all sentences related to this aspect been highlighted?
[Key Phrases] Have all key phrases related to this aspect been highlighted?

CONCISENESS EVALUATION:
The system output must be fully accurate, without any errors.

Rating Scale for Conciseness:
5 Stars: All content is relevant to this aspect
4 Stars: Most content is relevant to this aspect
3 Stars: Some content is relevant, some irrelevant
2 Stars: Most content is irrelevant to this aspect
1 Star: All content is irrelevant or contains errors

Evaluate conciseness by answering:
[Summary] Does the summary contain irrelevant/incorrect information?
[Sentences] Are there highlighted sentences irrelevant to this aspect?
[Key Phrases] Are there highlighted key phrases irrelevant to this aspect?

UNDERSTANDING THE INPUT:
- You will be shown the full text to evaluate
- You will also see currently highlighted elements, where words marked with ** are considered key elements
- Example: In "The study included **50** patients with **diabetes**", the words "50" and "diabetes" are marked as key elements
- Your task is to evaluate if these highlighted elements are appropriate and complete for the given aspect

You must ALWAYS respond with a valid JSON object using this exact structure:
{
    "ratings": {
        "com_eval_summary": <integer 1-5>,
        "con_eval_summary": <integer 1-5>,
        "com_eval_sentences": <integer 1-5>,
        "con_eval_sentences": <integer 1-5>,
        "com_eval_kps": <integer 1-5>,
        "con_eval_kps": <integer 1-5>
    },
    "explanation": {
        "ratings_rationale": "<string explaining your ratings, including any missing or incorrect elements>",
        "analysis_details": "<string providing detailed analysis of what information is missing or incorrectly highlighted>"
    }
}

IMPORTANT:
1. Do not include any text before or after the JSON
2. All rating values must be integers between 1 and 5
3. The response must be valid JSON that can be parsed
4. Do not include any markdown formatting or code blocks
5. In the explanation, be specific about any missing or incorrectly highlighted information"""

    # Prepare the evaluation context with existing summary and key elements if available
    evaluation_context = f"Aspect to evaluate: {aspect}\n\n"
    if aspect_data:
        evaluation_context += f"Existing summary: {aspect_data.get('summary', 'None')}\n\n"
        if 'kes' in aspect_data:
            evaluation_context += "Currently highlighted elements:\n"
            # Create a mapping of sentence indices to their key elements
            sentence_to_kes = {}
            for ke in aspect_data['kes']:
                s_idx = ke['sentence']
                if s_idx not in sentence_to_kes:
                    sentence_to_kes[s_idx] = []
                sentence_to_kes[s_idx].append(ke['index'])
            
            # For each sentence that has key elements, show the sentence with highlighted words
            for s_idx in sorted(sentence_to_kes.keys()):
                if 0 <= s_idx < len(article_tokens):
                    # Build the sentence with highlighted words marked with **
                    sentence_parts = []
                    for word_idx, word in enumerate(article_tokens[s_idx]):
                        if word_idx in sentence_to_kes[s_idx]:
                            sentence_parts.append(f"**{word}**")
                        else:
                            sentence_parts.append(word)
                    marked_sentence = ' '.join(sentence_parts)
                    evaluation_context += f"- {marked_sentence}\n"
        evaluation_context += "\n"

    user_message = f"""Evaluate the following medical text for the aspect of {aspect}.

{evaluation_context}
Text to evaluate:
{text}

Your task:
1. Evaluate how well the existing summary and key elements capture the relevant information from the text
2. Provide ratings (1-5) for completeness and conciseness of:
   - Summary (how well does the existing summary capture the aspect?)
   - Sentences (how well are the relevant sentences identified?)
   - Key phrases (how well are the key elements identified?)
3. Identify any missing or incorrect key elements
4. Provide detailed explanations for your ratings and key element selections

Remember: Your response must be ONLY a valid JSON object with no additional text."""

    request_data = {
        "model": st.session_state.selected_model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.1  # Lower temperature for more consistent JSON formatting
    }

    with debug_container:
        st.write("### API Request")
        st.json(request_data)

    try:
        with st.spinner("Making API request..."):
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            
            with debug_container:
                st.write("### Raw API Response")
                st.json(result)
            
            try:
                # Extract the JSON response from the message content
                response_content = result['choices'][0]['message']['content']
                
                # Clean the response content to ensure it's valid JSON
                response_content = response_content.strip()
                if response_content.startswith('```json'):
                    response_content = response_content[7:]
                if response_content.endswith('```'):
                    response_content = response_content[:-3]
                response_content = response_content.strip()
                
                with debug_container:
                    st.write("### Response Content")
                    st.code(response_content, language="json")
                
                # Parse the JSON response
                evaluation = json.loads(response_content)
                
                # Validate the required structure
                required_keys = ['ratings', 'explanation']
                required_ratings = [
                    'com_eval_summary', 'con_eval_summary',
                    'com_eval_sentences', 'con_eval_sentences',
                    'com_eval_kps', 'con_eval_kps'
                ]
                required_explanation = ['ratings_rationale', 'analysis_details']
                
                # Check for required top-level keys
                if not all(key in evaluation for key in required_keys):
                    missing = [key for key in required_keys if key not in evaluation]
                    raise ValueError(f"Missing required keys: {', '.join(missing)}")
                
                # Check for required rating keys
                if not all(key in evaluation['ratings'] for key in required_ratings):
                    missing = [key for key in required_ratings if key not in evaluation['ratings']]
                    raise ValueError(f"Missing required rating keys: {', '.join(missing)}")
                
                # Check for required explanation keys
                if not all(key in evaluation['explanation'] for key in required_explanation):
                    missing = [key for key in required_explanation if key not in evaluation['explanation']]
                    raise ValueError(f"Missing required explanation keys: {', '.join(missing)}")
                
                # Validate ratings are integers between 1 and 5
                for key, value in evaluation['ratings'].items():
                    if not isinstance(value, int) or value < 1 or value > 5:
                        raise ValueError(f"Invalid rating value for {key}: {value}. Must be integer between 1 and 5.")
                
                with debug_container:
                    st.write("### Parsed Evaluation")
                    st.json(evaluation)
                    
                    if 'explanation' in evaluation:
                        st.write("### AI Reasoning")
                        st.write("#### Rating Rationale")
                        st.write(evaluation['explanation']['ratings_rationale'])
                        st.write("#### Analysis Details")
                        st.write(evaluation['explanation']['analysis_details'])
                    
                    # Show token usage if available
                    if 'usage' in result:
                        st.write("### Token Usage")
                        st.json(result['usage'])
                
                return evaluation
                
            except json.JSONDecodeError as je:
                st.error(f"Invalid JSON structure: {je}")
                st.code(response_content)
                return None
            except ValueError as ve:
                st.error(f"Invalid response structure: {ve}")
                st.code(response_content)
                return None
            except KeyError as ke:
                st.error(f"Missing key in response: {ke}")
                st.json(result)
                return None
                
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling OpenRouter API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.write("Error response body:")
            st.code(e.response.text)
        return None

def main():
    st.title("Medical Article Evaluation Visualizer")
    
    # Add CSS styling for highlights
    st.markdown("""
        <style>
        .highlight-sentence {
            padding: 1px 3px;
            border-radius: 3px;
            margin: 0 1px;
            transition: background-color 0.2s;
        }
        .highlight-keyword {
            padding: 1px 3px;
            border-radius: 3px;
            margin: 0 1px;
            font-weight: 500;
            transition: background-color 0.2s;
        }
        .legend {
            margin-top: 30px;
            padding: 15px;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .legend h3 {
            margin-top: 0;
        }
        .legend-item {
            margin-bottom: 8px;
            display: flex;
            align-items: center;
        }
        .color-swatch {
            width: 20px;
            height: 20px;
            border: 1px solid #777;
            margin-right: 5px;
            display: inline-block;
            vertical-align: middle;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_json_data()
    if not data:
        return
    
    # Display article information
    article_data = data['articles']
    st.header("Article Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Title:** {article_data['title']}")
        st.write(f"**Author:** {article_data['author']}")
        st.write(f"**Journal:** {article_data['journal']}")
    
    with col2:
        st.write(f"**Year:** {article_data['year']}")
        st.write(f"**PMID:** {article_data['pmid']}")
    
    # Create aspect selector
    st.header("Aspect Analysis")
    
    # Convert summaries and article data from string to list if needed
    if isinstance(article_data['summaries'], str):
        summaries = json.loads(article_data['summaries'])
    else:
        summaries = article_data['summaries']
        
    if isinstance(article_data['article_tokens'], str):
        article_tokens = json.loads(article_data['article_tokens'])
    else:
        article_tokens = article_data['article_tokens']
        
    if isinstance(article_data['article'], str):
        article_text = json.loads(article_data['article'])
    else:
        article_text = article_data['article']
    
    # Create a dictionary of aspects and their summaries
    aspect_summaries = {summary['aspect']: summary for summary in summaries}
    
    # Create aspect selector
    selected_aspect = st.selectbox(
        "Select Aspect to Analyze",
        options=list(ASPECT_LABELS.keys()),
        format_func=lambda x: ASPECT_LABELS[x]
    )

    # Add model selector
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "meta-llama/llama-3.3-8b-instruct:free"

    st.selectbox(
        "Select AI Model",
        options=[
            "deepseek/deepseek-r1-0528:free",
            "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "meta-llama/llama-3.3-8b-instruct:free",
            "meta-llama/llama-4-maverick:free",
            "google/gemma-3-12b-it:free"
        ],
        key="selected_model"
    )
    
    # Add auto-evaluate button with debug option
    col1, col2 = st.columns([1, 3])
    with col1:
        auto_evaluate = st.button("🤖 Auto-Evaluate with AI")
    
    if auto_evaluate:
        # Join the article text for evaluation
        full_text = " ".join(article_text)
        
        # Get the selected aspect data
        selected_aspect_data = aspect_summaries.get(selected_aspect)
        
        # Get AI evaluation
        evaluation = evaluate_with_openrouter(
            full_text, 
            ASPECT_LABELS[selected_aspect], 
            selected_aspect_data,
            article_tokens
        )
        
        if evaluation:
            # Update the aspect data with AI evaluation
            if selected_aspect in aspect_summaries:
                aspect_data = aspect_summaries[selected_aspect]
                
                # Update only the ratings, not the key elements
                for key, value in evaluation['ratings'].items():
                    aspect_data[key] = value
                
                st.success("AI evaluation completed successfully!")
    
    # Display full article with highlighting
    st.subheader("Full Article Text")
    
    # Get the selected aspect data
    selected_aspect_data = aspect_summaries.get(selected_aspect)
    
    # Highlight the text for the selected aspect
    highlighted_article = highlight_article_text(
        article_text,
        article_tokens,
        selected_aspect_data
    )
    
    # Display the highlighted article
    st.markdown(highlighted_article, unsafe_allow_html=True)
    
    # Add legend for highlighting
    if selected_aspect in aspect_summaries:
        aspect_sentence_color, aspect_keyword_color = get_aspect_colors(selected_aspect)
        st.markdown(f"""
        <div class="legend">
            <h3>Highlighting Legend</h3>
            <div class="legend-item">
                <span class="color-swatch" style="background-color: {aspect_sentence_color};"></span>
                <span>Sentence containing key elements</span>
            </div>
            <div class="legend-item">
                <span class="color-swatch" style="background-color: {aspect_keyword_color};"></span>
                <span>Key element</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if selected_aspect in aspect_summaries:
        aspect_data = aspect_summaries[selected_aspect]
        
        # Display summary
        st.subheader(f"{ASPECT_LABELS[selected_aspect]} Summary")
        st.write(aspect_data['summary'])
        
        # Create and display radar chart
        radar_chart = create_evaluation_radar_chart(
            aspect_data,
            f"Evaluation Metrics for {ASPECT_LABELS[selected_aspect]}"
        )
        st.plotly_chart(radar_chart, use_container_width=True)
        
        # Create evaluation sections with star ratings
        st.header("Evaluation")
        
        # Completeness Evaluation
        completeness_values = {
            'summary': aspect_data['com_eval_summary'],
            'sentences': aspect_data['com_eval_sentences'],
            'kps': aspect_data['com_eval_kps']
        }
        completeness_ratings = create_rating_section(
            "Completeness Evaluation",
            "completeness",
            completeness_values
        )
        
        # Add some spacing
        st.write("")
        
        # Conciseness Evaluation
        conciseness_values = {
            'summary': aspect_data['con_eval_summary'],
            'sentences': aspect_data['con_eval_sentences'],
            'kps': aspect_data['con_eval_kps']
        }
        conciseness_ratings = create_rating_section(
            "Conciseness Evaluation",
            "conciseness",
            conciseness_values
        )

if __name__ == "__main__":
    main() 