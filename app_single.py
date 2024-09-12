from rag_system import RAGSystem
import time

rag_system = RAGSystem()

def main():
    start_time = time.time()
    question = "What is the signal word used for the product?"    
    question = "What are the first aid measures in case of swallowed?"
    question = "What is the signal word used for the product?"
    question = "What is the emergency telephone number for the product?"
    question = "What is the signal words?"
    
    # response = rag_system.answer_query(question)
    response = rag_system.answer_query(question)
    print(question)  # Convert cnt to string
    print(response + "\n--- %s seconds ---" % (time.time() - start_time))

# def main():
#     st.title("Ask me anything!")
#     question = st.text_input("Ask your question:")
#     if st.button("Ask"):
#         start_time = time.time()
#         response = rag_system.answer_query(question)
#         st.write(response + "\n--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
