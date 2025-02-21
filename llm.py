from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from fewshot_doc import answer_examples

chat_storage = {}
    
def get_llm():
    llm = ChatUpstage()
    return llm
    
# 사전을 참고하여 유저의 질문을 재구성하는 시스템 설정
def get_dictionary_chain():
    
    llm = get_llm()
    
    dictionary = [
        "베이스에 있는 선수를 나타내는 표현 -> 주자",
        "타석에 있는 선수를 나타내는 표현 -> 타자",
        "마운드 위에서 공을 던지는 선수를 나타내는 표현 -> 투수"
    ]
    
    dictionary_prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 확인하고 사전을 참고하여 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단되면 변경하지 않고 질문을 그대로 리턴해주세요.
        질문이나 주어가 모호할 경우 history retriever에서 history chat을 참고하여 유추할 수 있도록 질문을 변경하지 않고 그대로 리턴해주세요.
        사전: {dictionary}
        질문: {{question}}
                                                         """)
    
    dictionary_chain = dictionary_prompt | llm | StrOutputParser()
    return dictionary_chain
    
# 문서를 불러오기 위한 retriever 설정
def get_retriever():
    
    index_name = "baseball-rules-index"

    # 데이터를 벡터화할 임베딩 모델 설정
    embedding = UpstageEmbeddings(model="solar-embedding-1-large")
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={"k": 4})
    return retriever
    
# chat history를 반영하여 질문을 재구성하는 시스템 설정
def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    
    # chat history를 반영하여 질문을 재구성하는 시스템 프롬프트
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    # 채팅 기록을 반영하여 QA를 위한 프롬프트
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"), # 메시지 리스트를 전달
            ("human", "{input}")
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    return history_aware_retriever
    
# 세션 기록을 가져오기 위한 함수 설정
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # 세션이 없으면 해당 세션에 대한 채팅기록 생성
    if session_id not in chat_storage:
        chat_storage[session_id] = ChatMessageHistory()
    return chat_storage[session_id]
    
# RAG 체인을 생성
def get_rag_chain():
    
    llm = get_llm()
    
    # few-shot을 위한 프롬프트 설정
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}")
        ]
    )
    
    # few-shot을 위한 프롬프트 설정
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples
    )
    
    # 모델의 역할을 정의해주기 위한 시스템 프롬프트
    system_prompt = ("""
            당신은 최고의 KBO 리그 야구 규칙 전문가입니다. 
            아래의 문서를 참고해서 사용자의 질문에 대한 답변을 해주세요.
            
            먄약에 야구 규칙과 관련이 없다면 "🚫야구 규칙에 대한 질문만 답변이 가능합니다."라고 답변해주세요.
            만약에 답변할 수 없다면 모른다고 답변해주세요.
            
            답변 시에 출처에 대해서도 명확하게 밝혀주시고, 사용자가 이해하기 쉽게 설명해주세요.
            답변의 길이는 2-3줄 정도로 제한해주세요.
            \n\n
            문서: {context}
    """)
    
    # 채팅 기록 반영하여 QA를 위한 프롬프트
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt, # few-shot을 참고하도록 전달
            MessagesPlaceholder("chat_history"), # 메시지 리스트(채팅 기록)를 전달
            ("human", "{input}")
        ]
    )
    
    history_aware_retriever = get_history_retriever() # 기록 기반 검색을 위한 retriever
    qa_chain = create_stuff_documents_chain(llm, qa_prompt) # 모델에게 검색한 문서 리스트를 전달
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain) # RAG 체인 생성
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, # 실행할 RAG 체인이나 LLM 체인
        get_session_history, # 세션_ID 기반의 채팅 기록
        input_messages_key="input", # 유저의 입력이 담긴 키
        history_messages_key="chat_history", # 채팅 기록을 전달할 키
        output_messages_key="answer" # 출력 결과를 저장할 키
    ).pick("answer")
    
    # 채팅 history를 포함한 retriever를 활용하여 rag_chain 반환
    return conversational_rag_chain
    
def get_ai_response(query):
    
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    
    baseball_chain = {"input": dictionary_chain} | rag_chain
    
    ai_response = baseball_chain.stream(
        {"question": query},
        config={"configurable": {"session_id": "abc123"}}
    )
    
    return ai_response


# ai_response = get_ai_response()
# print(f"ai_response:\n {ai_response}\n")