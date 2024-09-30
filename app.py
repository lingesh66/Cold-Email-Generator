import streamlit as st
import uuid
import chromadb
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    api_key="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",
    model_name="llama-3.1-70b-versatile"
)

# Streamlit UI
st.title("Cold Email Generator")

# Input: Blank URL input
url = st.text_input("Enter the job posting URL")

if st.button("Generate Cold Email"):

    # Load web page content
    Loader = WebBaseLoader(url)
    page_data = Loader.load().pop().page_content

    # Extract job details from the page content
    prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the 
        following keys: `role`, `experience`, `skills` and `description`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):    
        """
    )
    
    chain_extract = prompt_extract | llm
    res = chain_extract.invoke(input={'page_data': page_data})

    json_parser = JsonOutputParser()
    job = json_parser.parse(res.content)

    # Load portfolio CSV and ChromaDB setup
    df = pd.read_csv("my_portfolio.csv")
    client = chromadb.PersistentClient('vectorstore')
    collection = client.get_or_create_collection(name="portfolio")

    if not collection.count():
        for _, row in df.iterrows():
            collection.add(documents=row["Techstack"],
                           metadatas={"links": row["Links"]},
                           ids=[str(uuid.uuid4())])

    # Query portfolio based on job skills
    links = collection.query(query_texts=job['skills'], n_results=2).get('metadatas', [])

    # Generate cold email
    prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        ### INSTRUCTION:
        You are lingesh, a business development executive at corer.Corer is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools. 
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
        process optimization, cost reduction, and heightened overall efficiency. 
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of corer 
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase corer's portfolio: {link_list}
        Remember you are lingesh, BDE at corer . 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        """
        
    )
    
    chain_email = prompt_email | llm
    email_res = chain_email.invoke(input={"job_description": str(job), "link_list": links})

    # Display only the generated cold email
    st.write("### Generated Cold Email:")
    st.text(email_res.content)


