import streamlit as st
import PyPDF2
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import warnings
import tempfile

# Download necessary NLTK data
import nltk
import os

nltk_data_path = os.path.expanduser('~/nltk_data')
nltk.data.path.append(nltk_data_path)

nltk.download('punkt_tab', download_dir=nltk_data_path)

from nltk.tokenize import sent_tokenize

warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

st.set_page_config(page_title="PDF Word2Vec Tool", layout="wide")

st.title("PDF Word2Vec Analyzer")
st.markdown("This tool allows you to upload a PDF file, analyze its content using Word2Vec, and explore similar words interactively.")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    
    # Extract text from the uploaded PDF
    pdf_text = ""
    with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    with open(temp_pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() or ""  # Handle NoneType

    if pdf_text.strip():
        st.subheader("Extracted Text")
        st.text_area("PDF Content", pdf_text, height=200)

        # Tokenize and preprocess text
        try:
            sentences = sent_tokenize(pdf_text)
            preprocessed_sentences = [simple_preprocess(sentence) for sentence in sentences if sentence.strip()]

            if preprocessed_sentences:
                # Train Word2Vec model
                model = Word2Vec(sentences=preprocessed_sentences, vector_size=100, window=5, min_count=1, workers=4)

                st.subheader("Word2Vec Model Training Complete")
                
                # Interactive section to explore Word2Vec
                word = st.text_input("Enter a word to find similar terms:")
                if word:
                    try:
                        similar_words = model.wv.most_similar(word)
                        st.write("Most similar words:")
                        for similar_word, similarity in similar_words:
                            st.write(f"{similar_word}: {similarity:.4f}")
                    except KeyError:
                        st.error("Word not found in vocabulary. Try another word.")
            else:
                st.error("No valid sentences were found after preprocessing.")
        
        except Exception as e:
            st.error(f"An error occurred while processing the text: {e}")

        # Visualization feature request placeholder
        st.markdown("---")
        st.info("Future updates will include data visualizations for word clusters!")

    else:
        st.error("No text could be extracted from the PDF. Ensure the PDF contains selectable text.")

else:
    st.warning("Please upload a PDF to begin.")

    

    
