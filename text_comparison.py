import streamlit as st

st.set_page_config(
        page_title="Compare Your Text",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )

slider_style = """
    <style>
        .css-1e8v41w {
            font-size: 20px; /* Adjust the font size here */
        }
    </style>
"""

hide_default_format = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)

from sentence_transformers import SentenceTransformer, util
from spacy.lang.en.stop_words import STOP_WORDS
from IPython.display import HTML
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

import spacy
import numpy as np
import string
import torch
import pandas as pd
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')


# Load the SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

# Sample text for Document 1 and Document 2
sample_documents = {
    'Agribusiness Fundamentals': {
        'Document 1': "Agribusiness/Agricultural Business Operations: A program that prepares individuals to manage agricultural businesses and agriculturally related operations within diversified corporations. Includes instruction in agriculture, agricultural specialization, business management, accounting, finance, marketing, planning, human resources management, and other managerial responsibilities.",
        'Document 2': "AAEC 5034: Agribusiness Marketing Policy and Business Strategies: Marketing tools needed to identify and solve the complexity of marketing food and agribusiness products. Contemporary trends, marketing strategies, and problems in the food and agribusiness sector. \nAAEC 5054: Strategic Agribusiness Management Application of economic theory to operational and strategic decision-making in agribusiness. Analysis and application of the functions of management. Problem recognition and economic analysis of supply chain, marketing, financial, production, and human resource decisions facing agribusiness firms. Assessment of U.S. role in the international marketplace. \n AAEC 5074: Agricultural and Food Policy: Policy issues related to trade, farm bills, natural resource preservation, and food, nutrition, and health. Global forces impacting U.S. policy. Stakeholder influence on the policy-making process. Policy impacts on stakeholders. \n AAEC 5424: Agribusiness Finance and Risk Management: Introduction to corporate finance and risk management in agribusiness. Financial analysis, estimation of capital cost and valuation. Focus on risk management and Environmental and Social Governance (ESG) practices through case studies."
    },
    'Bioethics': {
        'Document 1': "Bioethics/Medical Ethics: A program that focuses on the application of ethics, religion, jurisprudence, and the social sciences to the analysis of health care issues, clinical decision-making, and research procedures. Includes instruction in philosophical ethics, moral value, medical sociology, theology, spirituality and health, policy analysis, decision theory, and applications to problems such as death and dying, therapeutic relationships, organ transplantation, human and animal subjects, reproduction and fertility, health care justice, cultural sensitivity, needs assessment, professionalism, conflict of interest, chaplaincy, and clinical or emergency procedures.",
        'Document 2': "PHS 5724: Ethical Foundations of Public Health: Methods for ethics decision-making in public health and health policy, exploration of theoretical foundations of ethical public health practice, methods for identifying ethical challenges and ethical dilemmas, skills for managing ethical ambiguity, differences and similarities between professional ethics, research ethics, clinical ethics, and public health ethics, key historical events in public health that led to ethical and policy requirements, decision-making frameworks to analyze public health ethical challenges, current writings in public health ethics literature, well-reasoned written and oral arguments for a course of action to address public health ethics dilemmas. \n STS 5444: Issues in Bioethics: Identification and analysis of ethical issues arising in basic and applied biological, medical, environmental, ecological, and energy studies."
    },
    'Elementary Education': {
        'Document 1':"Elementary Education and Teaching: A program that prepares individuals to teach students in the elementary grades, which may include kindergarten through grade eight, depending on the school system or state regulations. Includes preparation to teach all elementary education subject matter.",
        'Document 2':"EDCI 5014: The 21st Century Classroom: Elementary Science Methods (PreK-6): Instructional and assessment approaches in elementary science methods. Includes assessment of learning, reflection on current research, implementation and integration of science content, inquiry-based science practices, and scientific investigation planning.\n EDCI 5024: The 21st Century Classroom: Elementary Math Methods (PreK-6): Instructional and assessment approaches in elementary math. Includes assessment of learning, reflection on current research, implementation and integration of math content for in person and online instruction, identification of children’s number sense and computational thinking, and how to incorporate the five processes of math instruction. \n EDCI 5034: The 21st Century Classroom: Literacy Methods in the Elem Classroom (PreK-6): Instructional and assessment approaches in elementary reading and language arts methods. Includes assessment of learning, reflection on current language and literacy research, developmental reading skills and language arts lessons for an in person and online instruction, and the interconnectedness of the language arts skills, and phonics instruction. \n EDCI 5044: The 21st Century Classroom: Content Literacy in a Global Society (PreK-6): Instructional and assessment approaches in content literacy integrated in social studies. Includes assessment of learning, reflection on content reading and elementary social studies methods research, culturally responsive activities/lessons, evaluate children’s literature as mentor texts, and design social studies lessons for a global society. \n EDCI 5054: The 21st Century Classroom: Elementary Curriculum & Instruction (PreK-6): Instructional and assessment approaches in the elementary classroom to meet the needs of all learners. Includes assessment of learning, reflection on current research, and teaching approaches for diverse learners, development and learning theories, Universal Design for Learning, and the professional, legal, and ethics of teaching."

    }
}
def main():
# Streamlit app
    # Create a two column layout
    col1, col2 = st.columns(2, gap='medium')

    with col1:

        st.image('https://www.vt.edu/content/vt_edu/en/admissions/undergraduate/visit/campus-photo-tour/jcr:content/content/adaptiveimage_1377849292.transform/xl-medium/image.jpg',
        use_column_width='always')

        st.title('Text Comparison Demo')
        st.markdown('Find out how the course descriptions of a program matches to its target CIP code descriptions with highlighted matching words.')
        with st.sidebar:

            st.title('Dive right in! 😃')
            # sidebar with sample document selection
            selected_sample = st.sidebar.selectbox('Select a sample to compare', list(sample_documents.keys()))
        # Set the initial values of document1 an document2 based on selected sample
        if selected_sample in sample_documents:
            document1_initial_value = sample_documents[selected_sample]['Document 1']
            document2_initial_value = sample_documents[selected_sample]['Document 2']
        else:
            document1_initial_value = ""
            document2_initial_value = ""


        # Add input field for similarity similarity_threshold
        st.markdown(slider_style, unsafe_allow_html=True)
        similarity_threshold = st.slider('🆒 Similarity Threshold', 0.0, 1.0, 0.6)

        # Text input for document 1
        document1 = st.text_area("📕 Enter CIP Code Descriptions:", document1_initial_value, height=150)
        # Text input for document 2
        document2 = st.text_area("📗 Enter Course Descriptions: (❗️Please start each course description on a new line, see the sample text for details):", document2_initial_value, height=350)

        if st.button('Compare'):
            with col2:
                # Process the documents and calculate similarities
                document1_embedding = model.encode([preprocess_text(document1)])[0]
                words_document1 = preprocess_text(document1).split()

                # Preprocess document 2
                document2_sentences = document2.split('\n')

                # Show similarity scores first
                    # Load different embedding models
                sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                phrase_model = SentenceTransformer('whaleloops/phrase-bert')

                    # Build the word emebdding model
                def word_model(doc1, doc2, vector_size=100,window=5, min_count=1, workers=4, epochs=20):
                    #preprocessing
                    stop_words = set(stopwords.words('english'))
                    def preprocess_word(text):
                        words = word_tokenize(text.lower())
                        words = [word for word in words if word.isalpha() and word not in stop_words]
                        return words
                    # tokenize the document
                    tokens_doc1 = preprocess_word(document1)
                    tokens_doc2 = preprocess_word(document2)
                    # train the model
                    word_model = Word2Vec(sentences=[tokens_doc1, tokens_doc2],
                    vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=1)
                    # calculate average vectors for each document
                    word_emb_doc1 = np.mean([word_model.wv[word] for word in tokens_doc1], axis=0)
                    word_emb_doc2 = np.mean([word_model.wv[word] for word in tokens_doc2], axis=0)
                    # calculate the similarity score
                    word_sim = round(float(cosine_similarity([word_emb_doc1], [word_emb_doc2])[0][0]),2)

                    return word_sim


                    # Calculate phrase embeddings for document 1 and document 2
                phrase_emb_doc1 = phrase_model.encode(document1)
                phrase_emb_doc2 = phrase_model.encode(document2)

                    # Calculate sentence embedding for document 1 and document 2
                sentence_emb_doc1 = sentence_model.encode(document1)
                sentence_emb_doc2 = sentence_model.encode(document2)

                    # Calculate cosine similarity between phrase/sentence embeddings
                phrase_sim = util.pytorch_cos_sim(phrase_emb_doc1, phrase_emb_doc2).item()
                sentence_sim = util.pytorch_cos_sim(sentence_emb_doc1, sentence_emb_doc2).item()
                word_sim = round(word_model(document1, document2),2)

                    # Display similarity scores
                st.subheader('🔢 Similarity Scores by Embedding Model')
                st.metric(label='Phrase Embedding', value=round(phrase_sim,2))
                st.metric(label='Sentence Embedding', value=round(sentence_sim,2))
                st.metric(label='Word Embedding', value = word_sim)

                # Start the matching process
                # Create a set to store words matching with any sentence in document 2
                matching_words_set = set()

                # Loop through each sentence in document 2
                for sentence in document2_sentences:
                    document2_embedding = model.encode([preprocess_text(sentence)])[0]
                    words_document2 = preprocess_text(sentence).split()

                    # Calculate word similarity between word embeddings and update matching words set
                    for word1 in words_document1:
                        for word2 in words_document2:
                            similarity_score = util.pytorch_cos_sim(torch.tensor(model.encode([word1])), torch.tensor(model.encode([word2]))).item()
                            if similarity_score >= similarity_threshold:
                                matching_words_set.add(word1)

                # Create the matching sentences from Document 1 with matching words highlighted
                matching_sentences_document1 = " ".join([f"<span style='background-color: #508590;'>{word}</span>" if word in matching_words_set else word for word in words_document1])

                # Display Document 1 with matching words highlighted
                st.subheader('🔡 Matching Results')
                st.markdown(f"<b>📕 CIP Code Descriptions with matching words:</b><br>{matching_sentences_document1}", unsafe_allow_html=True)
                st.markdown('<hr>', unsafe_allow_html=True)

                # Loop through each sentence in document 2
                for index, sentence in enumerate(document2_sentences, start=1):
                    document2_embedding = model.encode([preprocess_text(sentence)])[0]
                    words_document2 = preprocess_text(sentence).split()

                    # Calculate sentence similarity between document 2 sentence and document 1
                    similarity_score = util.pytorch_cos_sim(torch.tensor(document2_embedding), torch.tensor(document1_embedding)).item()

                    # Create highlighted text for sentence in document 2
                    highlighted_document2 = []

                    matching_indices = []

                    for j, word2 in enumerate(words_document2):
                        for i, word1 in enumerate(words_document1):
                            similarity_score = util.pytorch_cos_sim(torch.tensor(model.encode([word1])), torch.tensor(model.encode([word2]))).item()
                            if similarity_score >= similarity_threshold:
                                matching_indices.append(j)

                    for j, word in enumerate(words_document2):
                        if any(index == j for index in matching_indices):
                            highlighted_document2.append(f"<span style='background-color: #508590;'>{word}</span>")
                        else:
                            highlighted_document2.append(word)

                    # Join the highlighted words to form the highlighted sentence
                    highlighted_sentence2 = " ".join(highlighted_document2)

                    # Display the matching sentence from Document 2 with line breaks
                    st.markdown(f"<p><b>📗 Course {index} with matching words highlighted:</b><br>{highlighted_sentence2}</p>", unsafe_allow_html=True)


def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.text.lower() not in STOP_WORDS and token.text not in string.punctuation]
    return " ".join(tokens)

if __name__ == "__main__":
    main()
