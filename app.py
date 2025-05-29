import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import re
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Medical Article Evaluation Visualizer",
    page_icon="🏥",
    layout="wide"
)

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

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

# Define colors for different aspects
ASPECT_COLORS = {
    'ob': '#DC3545',  # Red
    'i': '#007BFF',   # Blue
    'c': '#28A745',   # Green
    'b': '#FFC107',   # Yellow
    'p': '#17A2B8',   # Cyan
    'm': '#6610F2',   # Purple
    'td': '#FD7E14',  # Orange
    'pe': '#20C997',  # Teal
    'fd': '#E83E8C',  # Pink
    'o': '#6F42C1',   # Indigo
    'f': '#D39E00',   # Dark Yellow
    'rf': '#1E7E34'   # Dark Green
}

# Define star rating options
STAR_RATINGS = {
    1: "⭐",
    2: "⭐⭐",
    3: "⭐⭐⭐",
    4: "⭐⭐⭐⭐",
    5: "⭐⭐⭐⭐⭐"
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
    
    # Create columns for the three rating categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        summary_rating = st.selectbox(
            "Summary",
            options=list(STAR_RATINGS.keys()),
            format_func=lambda x: STAR_RATINGS[x],
            key=f"{prefix}_summary",
            index=initial_values['summary']-1
        )
        
    with col2:
        sentences_rating = st.selectbox(
            "Sentences",
            options=list(STAR_RATINGS.keys()),
            format_func=lambda x: STAR_RATINGS[x],
            key=f"{prefix}_sentences",
            index=initial_values['sentences']-1
        )
        
    with col3:
        kps_rating = st.selectbox(
            "Key Phrases",
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
    if not selected_aspect_data or 'kes' not in selected_aspect_data:
        return article_text
    
    # Get unique sentence numbers from the key elements
    sentence_numbers = {ke['sentence'] for ke in selected_aspect_data['kes']}
    
    # Create a dictionary to store words to highlight for each sentence
    highlight_words = {sent_num: set() for sent_num in sentence_numbers}
    
    # Collect words to highlight for each sentence
    for ke in selected_aspect_data['kes']:
        sent_num = ke['sentence']
        if sent_num < len(article_tokens):
            sentence = article_tokens[sent_num]
            if ke['index'] < len(sentence):
                highlight_words[sent_num].add(sentence[ke['index']])
    
    # Process the article text sentence by sentence
    highlighted_sentences = []
    for i, sentence in enumerate(article_text):
        if i in sentence_numbers:
            # Wrap the entire sentence in a darker background
            sentence_start = '<span style="background-color: #f0f0f0; padding: 0.2em 0;">'
            
            # Highlight specific words within the sentence
            words = sentence.split()
            highlighted_words = []
            for word in words:
                # Remove punctuation for comparison but keep it for display
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word in highlight_words[i]:
                    highlighted_words.append(f'<span style="background-color: #ffeb3b;">{word}</span>')
                else:
                    highlighted_words.append(word)
            
            # Join the words and close the sentence span
            highlighted_sentence = f"{sentence_start}{' '.join(highlighted_words)}</span>"
            highlighted_sentences.append(highlighted_sentence)
        else:
            highlighted_sentences.append(sentence)
    
    return ' '.join(highlighted_sentences)

def evaluate_with_openrouter(text, aspect):
    """
    Use OpenRouter API to evaluate the text for a specific aspect.
    """
    # Create a placeholder for the debug information
    debug_container = st.expander("🔍 Debug Information", expanded=True)
    
    with debug_container:
        st.write("### API Configuration")
        st.json({
            "API_URL": OPENROUTER_URL,
            "Model": "meta-llama/llama-3.3-8b-instruc:free",
            "Has_API_Key": bool(OPENROUTER_API_KEY)
        })

    if not OPENROUTER_API_KEY:
        st.error("OpenRouter API key not found. Please set it in the .env file.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:8501",
    }

    prompt = f"""Evaluate the following medical text for the aspect of {aspect}.
    Provide ratings on a scale of 1-5 for:
    1. Summary completeness
    2. Summary conciseness
    3. Sentence completeness
    4. Sentence conciseness
    5. Key phrases completeness
    6. Key phrases conciseness

    Also identify the key sentences and phrases that are relevant to this aspect.

    Text:
    {text}

    Only respond in format JSON with the following structure and DO NOT include any other text Before or After the JSON:
    {{
        "ratings": {{
            "com_eval_summary": number,
            "con_eval_summary": number,
            "com_eval_sentences": number,
            "con_eval_sentences": number,
            "com_eval_kps": number,
            "con_eval_kps": number
        }},
        "key_elements": [
            {{"sentence": number, "index": number}}
        ],
        "explanation": {{
            "ratings_rationale": string,
            "key_elements_selection": string
        }}
    }}
    
    Include detailed explanations in the explanation field for why you chose each rating and how you selected the key elements.
    """

    with debug_container:
        st.write("### Prompt")
        st.code(prompt, language="text")
        
        st.write("### API Request")
        request_data = {
            "model": "meta-llama/llama-3.3-8b-instruct:free",
            "messages": [{"role": "user", "content": prompt}]
        }
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
            
            # Extract the JSON response from the message content
            try:
                response_content = result['choices'][0]['message']['content']
                with debug_container:
                    st.write("### Response Content")
                    st.code(response_content, language="json")
                
                evaluation = json.loads(response_content)
                
                with debug_container:
                    st.write("### Parsed Evaluation")
                    st.json(evaluation)
                    
                    if 'explanation' in evaluation:
                        st.write("### AI Reasoning")
                        st.write("#### Rating Rationale")
                        st.write(evaluation['explanation']['ratings_rationale'])
                        st.write("#### Key Elements Selection Process")
                        st.write(evaluation['explanation']['key_elements_selection'])
                    
                    # Show token usage if available
                    if 'usage' in result:
                        st.write("### Token Usage")
                        st.json(result['usage'])
                
                return evaluation
                
            except json.JSONDecodeError as e:
                with debug_container:
                    st.error("### JSON Parse Error")
                    st.error(f"Error parsing response: {e}")
                    st.write("Raw content that failed to parse:")
                    st.code(response_content)
                return None
                
            except KeyError as e:
                with debug_container:
                    st.error("### Response Structure Error")
                    st.error(f"Missing key in response: {e}")
                    st.json(result)
                return None
                
    except requests.exceptions.RequestException as e:
        with debug_container:
            st.error("### API Request Error")
            st.error(f"Error calling OpenRouter API: {e}")
            if hasattr(e.response, 'text'):
                st.write("Error response body:")
                st.code(e.response.text)
        return None

def main():
    st.title("Medical Article Evaluation Visualizer")
    
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
    
    # Add auto-evaluate button with debug option
    col1, col2 = st.columns([1, 3])
    with col1:
        auto_evaluate = st.button("🤖 Auto-Evaluate with AI")
    
    if auto_evaluate:
        # Join the article text for evaluation
        full_text = " ".join(article_text)
        
        # Get AI evaluation
        evaluation = evaluate_with_openrouter(full_text, ASPECT_LABELS[selected_aspect])
        
        if evaluation:
            # Update the aspect data with AI evaluation
            if selected_aspect in aspect_summaries:
                aspect_data = aspect_summaries[selected_aspect]
                
                # Update ratings
                for key, value in evaluation['ratings'].items():
                    aspect_data[key] = value
                
                # Update key elements
                aspect_data['kes'] = evaluation['key_elements']
                
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
    st.markdown("""
    <div style="margin-top: 1em; padding: 1em; border-radius: 0.5em; background-color: #f0f2f6;">
        <p style="margin: 0;"><strong>Legend:</strong> 
        Sentences with a <span style="background-color: #f0f0f0; padding: 0.2em 0.4em;">light gray background</span> 
        contain key elements for the selected aspect. Within these sentences, specific key words are highlighted in 
        <span style="background-color: #ffeb3b;">yellow</span>.</p>
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