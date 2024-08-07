import streamlit as st
from rag_system import RAGSystem
import time

rag_system = RAGSystem()
# st.title("RAG-based AMA System")

def main():
    st.title("Ask me anything!")
    question = st.text_input("Ask your question:")
    if st.button("Ask"):
        start_time = time.time()
        response = rag_system.answer_query(question)
        st.write(response + "\n--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
