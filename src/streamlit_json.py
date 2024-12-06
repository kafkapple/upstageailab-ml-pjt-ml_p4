import streamlit as st
import requests as req

st.title("Upstage 4조 MLOps 프로젝트")
st.header('Streamlit')
user_input = st.text_input('감정분석 할 텍스트 입력')

if st.button('post'):
    if user_input.strip():
        try:
            r = req.post(
                "http://127.0.0.1:8000/infer",
                json={"infer": user_input}
            )
            if r.status_code == 200:
                result = r.json()
                st.text("당신의 지금 감정은? :")
                st.text(result['result'])
            else:
                st.text("FastAPI 서버에서 오류가 발생했습니다.")
                st.text(r.status_code)
        except req.exceptions.RequestException as e:
            st.error(f"Request failed: {str(e)}")
    else: 
        st.text("감정분석할 내용을 입력해주세요!")
        
if st.button('get'):
    st.text(req.get("http://127.0.0.1:8000/").text)

# st.components.v1.html(
#     """
#     <iframe src="https://danielinjesus.github.io/tosspayments/toss_blog_origin.html" width="100%" height="600" frameborder="0">
#     </iframe>
#     """,
#     height=600,
# )

#실행: streamlit run C:\Code_test\Twitter\m1_Streamlit\v1_streamlit_json.py