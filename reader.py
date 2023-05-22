# Import necessary classes and functions
from langchain.tools import BaseTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import Field
from langchain.chains.qa_with_sources.loading import BaseCombineDocumentsChain
import os, asyncio, trafilatura
from langchain.docstore.document import Document

# Function to return an instance of RecursiveCharacterTextSplitter with the specified parameters
def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
    )

# Define a class WebpageQATool that inherits from BaseTool
class WebpageQATool(BaseTool):
    name = "query_webpage"  # Name of the tool
    description = "Browse a webpage and retrieve the information relevant to the question."  # Description of the tool
    text_splitter: RecursiveCharacterTextSplitter = Field(default_factory=_get_text_splitter)  # Instance of RecursiveCharacterTextSplitter
    qa_chain: BaseCombineDocumentsChain  # Instance of BaseCombineDocumentsChain
    
    # Define a method to run the tool
    def _run(self, data: str) -> str:
        # Split the input data into question and url
        datalist = data.split(",")
        question, url = datalist[0], datalist[1]
        
        # Extract the content of the webpage
        result = trafilatura.extract(trafilatura.fetch_url(url))
        
        # Convert the content into Document objects
        docs = [Document(page_content=result, metadata={"source": url})]
        
        # Split the documents into smaller pieces
        web_docs = self.text_splitter.split_documents(docs)
        
        results = []
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i:i+4]
            # Run the QA chain on each window of documents
            window_result = self.qa_chain({"input_documents": input_docs, "question": question}, return_only_outputs=True)
            results.append(f"Response from window {i} - {window_result}")
            
        # Convert the results into Document objects
        results_docs = [Document(page_content="\n".join(results), metadata={"source": url})]
        
        # Run the QA chain on the results documents
        return self.qa_chain({"input_documents": results_docs, "question": question}, return_only_outputs=True)
    
    # Define an async method to run the tool
    async def _arun(self, url: str, question: str) -> str:
        # This method is not implemented
        raise NotImplementedError
