from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples
from dotenv import load_dotenv
load_dotenv()

store = {} # Store for storing the context of the conversation

# 현재 세션의 대화 기록을 가져오는 함수
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_llm():
    return ChatUpstage()


def get_retriever():
    embedding = UpstageEmbeddings(model="solar-embedding-1-large")
    index_name = "tax-table-index"
    database = PineconeVectorStore.from_existing_index(embedding=embedding, index_name=index_name)
    
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    
    # 유저가 질문하는 내용을 이해하고, 이전 대화를 참고하여 새로운 질문을 생성하는 프롬프트
    contextualize_q_system_prompt = (
        """
        Given a chat history and latest user question 
        which might reference context in the chat history, 
        formulate a standalone question that can be understood 
        without the chat history. DO NOT ANSWER THE QUESTION, 
        just formulate it if needed and otherwise return it as is.
        """
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # 기록 기반 검색 위한 retriever 생성
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    return history_aware_retriever
    
    
def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다. 그럴 경우에는 질문만 리턴해주세요.
        사전: {dictionary}
        
        질문: {{question}}
                                """)
    
    llm = get_llm()
    
    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain


def get_rag_chain():

    llm = get_llm()
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answerf}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    
    system_prompt = (
        """
        당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요.
        아래에 제공된 문서를 활용해서 답변해 주시고
        답변을 알 수 없다면 모른다고 답변해주세요. 
        답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고
        2-3 문장 정도의 짧은 답변을 원합니다.
        \n\n
        {context}
        """
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        
        [
            ("system", system_prompt), # LLM의 역할
            few_shot_prompt, # 처음에는 chat_history가 없는데 few_shot_prompt을 이전 대화로 인식하도록 설정
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # 질문-답변 체인 생성
    history_aware_retriever = get_history_retriever() # 기록 기반 검색 위한 retriever
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, 
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")
    return conversational_rag_chain # 채팅 history를 포함한 retriever를 활용해서 rag_chain 반환


def get_ai_response(user_message):
    
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    
    tax_chain = {"input": dictionary_chain} | rag_chain
    ai_response = tax_chain.stream(
        {"question": user_message},
        config={
            "configurable": {"session_id": "abc123"}
            },
        )
    return ai_response
    # return ai_message["answer"]
