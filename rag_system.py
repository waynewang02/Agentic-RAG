from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings import OllamaEmbeddings
# from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
import re

# pip install -U langchain-chroma
class RAGSystem:
    def __init__(self, data_dir_path = "pdf", db_path = "chroma") -> None:
        # print("inside rag system")
        self.data_directory = data_dir_path
        self.db_path = db_path
        self.model_name = "llama3.1"
        self.llm_model = "llama3.1"
        self._setup_collection() 
        self.model = Ollama(model=self.llm_model)
        self.prompt_template = """
            Answer the question based on the following context: {context}.

            ---
            Answer the question: {question}. First, search for the context title.

            Reply in the format: 
            {{"answer": "your_answer_here", "source": "your_section_title_here"}}

            If the answer contains multiple parts, combine them into a single answer and reply in the format:
            {{"answer": "answer1, answer2, answer3, etc.", "source": "source1, source2, source3, etc."}}

            If you do not know the answer, reply:
            {{"answer": "I don't know", "source": "N/A"}}

            Do not provide any additional notes or suggestions.
        """        
        # print(self.prompt_template)

    def _clean_text(self, text):
        """ Clean the document text by removing unnecessary headers, footers, and formatting characters. """
        # Remove headers and footers using regex
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)  # Remove "Page X of Y"
        text = re.sub(r'_{10,}', '', text)  # Remove long sequences of underscores (formatting characters)
        
        # Remove multiple spaces, tabs, and newline characters
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize white spaces

        return text

    def _setup_collection(self):
        pages = self._load_documents()
        chunks = self._document_splitter(pages)
        chunks = self._get_chunk_ids(chunks)
        vectordb = self._initialize_vectorDB()
        present_in_db = vectordb.get()
        ids_in_db = present_in_db["ids"]
        print(f"Number of existing ids in db: {len(ids_in_db)}")
        # add chunks to db - check if they already exist
        chunks_to_add = [i for i in chunks if i.metadata.get("chunk_id") not in ids_in_db]
        if len(chunks_to_add) > 0:
            vectordb.add_documents(chunks_to_add, ids = [i.metadata["chunk_id"] for i in chunks_to_add])
            print(f"added to db: {len(chunks_to_add)} records")
            # vectordb.persist()
        else:
            print("No records to add")

    def _get_chunk_ids(self, chunks):
        ''''
        for same page number: x
            source_x_0
            source_x_1
            source_x_2
        for same source but page number: x+1
            source_x+1_0
            source_x+1_1
            source_x+1_2
        '''
        prev_page_id = None
        for i in chunks:
            src = i.metadata.get("source")
            page = i.metadata.get("page")
            curr_page_id = f"{src}_{page}"
            if curr_page_id == prev_page_id:
                curr_chunk_index += 1
            else:
                curr_chunk_index = 0
            # final id of chunk
            curr_chunk_id = f"{curr_page_id}_{curr_chunk_index}"
            prev_page_id = curr_page_id
            i.metadata["chunk_id"] = curr_chunk_id
        return chunks        
    
    def _retrieve_context_from_query(self, query_text):
        vectordb = self._initialize_vectorDB()
        context = vectordb.similarity_search_with_score(query_text, k=4)
        return context
    
    def _get_prompt(self, query_text):
        context = self._retrieve_context_from_query(query_text)
        # print(f" ***** CONTEXT ******{context} \n")
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in context])
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        prompt = prompt_template.format(context=context_text, question=query_text)
        return prompt

    def answer_query(self, query_text):
        prompt = self._get_prompt(query_text)
        response_text = self.model.invoke(prompt)
        formatted_response = f"{response_text}\n"
        return formatted_response

    def _load_documents(self):
        loader = PyPDFDirectoryLoader(self.data_directory)
        pages = loader.load()
        # Clean the content of each page
        for page in pages:
            page.page_content = self._clean_text(page.page_content)
        
        return pages
        
    def _document_splitter(self, documents):
        def word_count(documents):
            return len(documents.split())
        splitter = RecursiveCharacterTextSplitter(
                # chunk_size=400,
                # chunk_overlap=100,
                # length_function=len,
                chunk_size=2000,
                chunk_overlap=500,
                length_function=len,
                is_separator_regex=False,
        )

        # Split the documents into chunks
        chunks = splitter.split_documents(documents)
        from datetime import datetime
        current_time = datetime.now().strftime('%m%d %I%M%p').lower()
        filename = f'{current_time}_chunks.txt'
            
        # Print each chunk with its metadata
        # for idx, chunk in enumerate(chunks):
        #     print(f"Chunk {idx + 1}:")
        #     print(f"Content: {chunk.page_content}")
        #     print(f"Metadata: {chunk.metadata}")
        #     print("\n" + "-"*80 + "\n")  # Separator for readability
        #     with open(filename, 'w') as file:
        #         file.write(f"Chunk {idx + 1}:\n")
        #         file.write(f"Content: {chunk.page_content}\n")
        #         file.write(f"Metadata: {chunk.metadata}")
        #         file.write("\n" + "-"*80 + "\n")

        with open(filename, 'a') as file:
            # Print each chunk with its metadata
            for idx, chunk in enumerate(chunks):
                print(f"Chunk {idx + 1}:")
                print(f"Content: {chunk.page_content}")
                print(f"Metadata: {chunk.metadata}")
                print("\n" + "-" * 80 + "\n")  # Separator for readability

                # Write chunk information to the file in append mode
                file.write(f"Chunk {idx + 1}:\n")
                file.write(f"Content: {chunk.page_content}\n")
                file.write(f"Metadata: {chunk.metadata}\n")
                file.write("\n" + "-" * 80 + "\n")
        
        return chunks
    
    def _get_embedding_func(self):
        embeddings = OllamaEmbeddings(model=self.model_name)
        return embeddings
    
    def _initialize_vectorDB(self):
        return Chroma(
            persist_directory = self.db_path,
            embedding_function = self._get_embedding_func(),
        )