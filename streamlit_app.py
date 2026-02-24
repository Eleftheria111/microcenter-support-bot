import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from app.rag import ask

st.set_page_config(page_title='Microcenter Support', page_icon='🤖')
st.title('🤖 Microcenter.gr Customer Support')
st.caption('Ρώτησέ με για προϊόντα, αποστολή, επιστροφές, ή ό,τι άλλο χρειάζεσαι!')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

if prompt := st.chat_input('Γράψε την ερώτησή σου...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.write(prompt)

    # Convert stored messages to LangChain message objects (exclude current message)
    history = []
    for msg in st.session_state.messages[:-1]:
        if msg['role'] == 'user':
            history.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            history.append(AIMessage(content=msg['content']))

    with st.chat_message('assistant'):
        with st.spinner('Σκέφτομαι...'):
            response = ask(prompt, chat_history=history)
        st.write(response['answer'])

    st.session_state.messages.append({
        'role': 'assistant',
        'content': response['answer'],
    })
