import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import PyPDF2
import argparse
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

import requests
from fpdf import FPDF
from bs4 import BeautifulSoup
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def crawl_webpage(url):
    response = requests.get(url)
    print(f"Response Status Code: {response.status_code} for {url}")
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = ' '.join([para.get_text() for para in paragraphs])
        return text_content
    else:
        return None

# Ensure the OpenAI API key is set
CHROMA_PATH = "chroma"

if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)
DATA_PATH = "data"

# List of websites to crawl
urls = [
    'https://www.techtarget.com/whatis/definition/pseudoscience#:~:text=The%20term%20pseudoscience%20can%20refer,stand%20up%20under%20scientific%20scrutiny.',
    "https://astrologify.com/zodiac/signs/",
    "https://atlasmythica.com/superstitions/",
    "https://www.yourtango.com/self/common-superstitions-from-around-world-people-believe",
    "https://www.livescience.com/ESP",
    "https://en.wikipedia.org/wiki/Reflexology#:~:text=It%20is%20based%20on%20a,related%20areas%20of%20the%20body."
]

def save_to_pdf(text, filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    encoded_text = text.encode('utf-8', 'replace').decode('latin-1')
    pdf.add_page()
    pdf.multi_cell(0, 10, encoded_text)
    pdf.output(filename)
    
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks
    
def get_embedding_function():
    return OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

def add_url_and_pdf_input():
    st.subheader("Add URLs and PDF Files")
    st.write("(Maximum 2 for each.)")

    # Limit to 2 URLs
    url1 = st.text_input("Enter URL 1:")
    url2 = st.text_input("Enter URL 2:")

    # Limit to 2 PDF uploads
    pdf1 = st.file_uploader("Upload PDF 1", type="pdf", label_visibility='collapsed')
    pdf2 = st.file_uploader("Upload PDF 2", type="pdf", label_visibility='collapsed')

    if st.button("Submit"):
        urls = []
        if url1:
            urls.append(url1)
        if url2:
            urls.append(url2)

        pdf_files = []
        if pdf1:
            pdf_files.append(pdf1)
        if pdf2:
            pdf_files.append(pdf2)

        # Process URLs
        for i, url in enumerate(urls):
            website_text = crawl_webpage(url)
            if website_text:
                filename = os.path.join("data/", f'url_content_{i+1}.pdf')

                # Save only if the PDF with that name doesn't already exist
                if not os.path.exists(filename):
                    save_to_pdf(website_text, filename)  # Save content to PDF

                    # Create a Document object to add to Chroma
                    doc = Document(page_content=website_text, metadata={"source": url})
                    add_to_chroma([doc])  # Add document to Chroma

                    st.success(f"Content from {url} saved to {filename} and added to the database.")
                else:
                    st.warning(f"File {filename} already exists. Content not saved to avoid overwriting.")
            else:
                st.error(f"Failed to retrieve content from {url}")


        # Process PDFs
        for i, pdf in enumerate(pdf_files):
            # Save the uploaded PDF
            pdf_path = os.path.join("data/", f'uploaded_pdf_{i+1}.pdf')
            if not os.path.exists(pdf_path):  # Check if the PDF already exists
                with open(pdf_path, "wb") as f:
                    f.write(pdf.getbuffer())
                st.success(f"Uploaded PDF saved to {pdf_path}")

                # Extract text from the uploaded PDF
                pdf_text = extract_text_from_pdf(pdf_path)

                # Create a Document object to add to Chroma
                doc = Document(page_content=pdf_text, metadata={"source": pdf_path})
                add_to_chroma([doc])  # Add document to Chroma

                st.success(f"Content from {pdf_path} added to the database.")

            else:
                st.warning(f"File {pdf_path} already exists. Upload not saved to avoid overwriting.")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"  # Extract text from each page
    return text

# Call the function to add URL and PDF input
add_url_and_pdf_input()

# Set up the OpenAI API key
my_secret = os.environ['OPENAI_API_KEY']

#set the app title
st.title('PseudoScience app')

st.write('Welcome to PseudoScience app. Ask me anything!')  # Initial text

# User inputs the question
question = st.text_input("Enter your questions:")

if st.button("Enter"):
    CHROMA_PATH = "chroma"
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    def main(query_text=question):
        # Prepare the embedding function.
        embedding_function = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

        # Load the Chroma DB
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Run similarity search
        results = db.similarity_search_with_relevance_scores(query_text, k=2)
        if len(results) == 0 or results[0][1] < 0.7:
            st.write("Unable to find matching results.")
            return

        # Create a context from the results
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        # Prepare the prompt with the context and the query
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Use OpenAI model to get a response
        model = ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'])
        response_text = model.predict(prompt)

        # Display the sources and the response
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        st.write(formatted_response)

    # Run the main function
    main()

def main2():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args(args=[])
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main2()
