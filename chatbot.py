import streamlit as st
from llm import get_ai_response

st.set_page_config(page_title="야구 규칙 챗봇", layout="wide")
st.title("⚾️야구 규칙 챗봇")
st.caption("KBO 야구 규칙에 대한 모든 것을 물어보세요!")

# session_state에 챗봇과 나눈 채팅 기록이 존재하지 않으면 저장하는 빈 배열 생성
if "message_list" not in st.session_state:
    st.session_state.message_list = []
    
# 유저와 챗봇의 채팅 기록을 모두 출력
for chat in st.session_state.message_list:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])
    
# 사용자의 입력 받기
if user_question := st.chat_input(placeholder="질문을 입력해 주세요!"):
    
    # 채팅이 "user"면 질문을 출력
    with st.chat_message("user"):
        st.write(user_question)
    # 유저의 입력 message_list에 추가
    st.session_state.message_list.append({"role": "user", "content": user_question})
    
    # 챗봇의 답변 생성
    with st.spinner("잠시만 기다려 주세요... 답변을 생성하는 중입니다..."):
        ai_response = get_ai_response(user_question)
        
        # 채팅이 "ai"면 답변을 출력
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            # 챗봇의 답변 message_list에 추가
            st.session_state.message_list.append({"role": "ai", "content": ai_message})