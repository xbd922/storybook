        import requests

        from fpdf import FPDF
        from bs4 import BeautifulSoup

        from openai import OpenAI

        import getpass
        import os

        import streamlit as st

        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role":
                    "system",
                    "content":
                    "You are an assistant that answers questions or statement related to pseudoscience only, and say you don't know if it is other topics."
                },  #system instruction
                {
                    "role": "user",
                    "content": "The New Age of Wellness: Is It Really Healthy?"
                },  #user input
            ],
        )

        print(response)


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


        # List of websites to crawl
        urls = [
            'https://medium.com/@milverton.saint/the-alkaline-diet-hoax-debunking-the-ph-myth-and-unraveling-the-truth-about-robert-o-young-e4495587c0a0',
            'https://en.wikipedia.org/wiki/BTS',
        ]

        # Crawl each URL
        # for url in urls:
        # website_text = crawl_webpage(url)
        # print(f"Extracted content from {url}:\n{website_text[:500]}...\n")  # Print first 500 chars of content


        def save_to_pdf(text, filename):
            # Create a PDF instance
            pdf = FPDF()

            pdf.set_auto_page_break(auto=True, margin=15)

            # Set font
            pdf.set_font("Arial", size=12)

            # Example text stored in a variable
            my_text = website_text

            # Encode the text to UTF-8 and then decode it to Latin-1, replacing unencodable characters
            # with a replacement character (e.g., �)
            encoded_text = my_text.encode('utf-8', 'replace').decode('latin-1')

            # Add a page
            pdf.add_page()

            # Add the encoded text to the PDF using the variable
            pdf.multi_cell(0, 10, encoded_text)

            path = "data/"
            full_path = os.path.join(path,filename)

            # Save the PDF
            pdf.output(full_path)


        for i, url in enumerate(urls):
            website_text = crawl_webpage(url)
            if website_text:
                # Save the extracted text to a PDF
                filename = f'website_content_{i+1}.pdf'
                save_to_pdf(website_text, filename)
                print(f"Content from {url} saved to {filename}")
            else:
                print(f"Failed to retrieve content from {url}")

        DATA_PATH = "/content/text_from_variable.pdf"


        def load_document2():
            document_loader = PyPDFLoader(DATA_PATH)
            return document_loader.load()