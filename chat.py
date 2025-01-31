import streamlit as st
from llm import get_ai_response

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

st.title("💸소득세 챗봇")
st.caption("소득세에 관한 모든 것을 물어보세요!")

# message_list: 유저가 입력한 채팅 값을 저장하는 array
if "message_list" not in st.session_state:
    st.session_state.message_list = []

# 이전에 유저가 작성했던 메시지 모두 출력
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 채팅 입력 & 입력한 채팅 출력 & message_list에 저장
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    # 챗봇 답변 저장 및 화면 출력
    with st.spinner("잠시만 기다려주세요... 답변을 생성하는 중입니다."):
        ai_response = get_ai_response(user_question)
        
        with st.chat_message("ai"):
            aim_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": aim_message})