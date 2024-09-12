from rag_system import RAGSystem
import time

rag_system = RAGSystem()
# st.title("RAG-based AMA System")

def main():
    
    questions = [
        "What is the signal word used for the product?",
        "What is the product identifier?",
        "When was the Safety Data Sheet issued?",
        "What is the product description?",
        "Who is the manufacturer of the product?",
        "What is the emergency telephone number for the product?",
        "What are the hazard classifications associated with the product?",        
        "What are the hazard statements related to the product?",
        "What precautions should be taken while handling the product?",
        "What are the first aid measures in case of inhalation?",
        "What are the first aid measures in case of skin contact?",
        "What are the first aid measures in case of eye contact?",
        "What are the first aid measures in case of ingestion?"
    ]
        
    start_time = time.time()
    from datetime import datetime
    current_time = datetime.now().strftime('%m%d %I%M%p').lower()
    filename = f'{current_time}.txt'

    cnt = 1
    with open(filename, 'w') as file:
        for question in questions:
            # loop_start_time = time.time()
            response = rag_system.answer_query(question)
            file.write(f"{cnt}: {question}\n")
            file.write(f"{response}\n")
            print(str(cnt) + ": " + question)  # Convert cnt to string
            print(response)
            cnt+=1
            
    print("\n--- Total %s seconds ---" % (time.time() - start_time))
# def main():
#     st.title("Ask me anything!")
#     question = st.text_input("Ask your question:")
#     if st.button("Ask"):
#         start_time = time.time()
#         response = rag_system.answer_query(question)
#         st.write(response + "\n--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
