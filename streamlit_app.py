import streamlit as st
from app.rag import ask

st.set_page_config(page_title='Microcenter Support', page_icon='🤖')
st.title('🤖 Microcenter.gr Customer Support')
st.caption('Ρώτησέ με οτιδήποτε για παραγγελίες, αποστολή, επιστροφές!')

@st.cache_resource
def load_chain():
    from app.rag import get_qa_chain
    return get_qa_chain()

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

if prompt := st.chat_input('Γράψε την ερώτησή σου...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.write(prompt)
    with st.chat_message('assistant'):
        with st.spinner('Σκέφτομαι...'):
            response = ask(prompt)
        st.write(response['answer'])
    st.session_state.messages.append({
        'role': 'assistant',
        'content': response['answer']
    })