from llms import * 
from langchain.chains import LLMChain


def main_program(article): 
    # article = input("Enter your story: ")  
    text_llm_obj = TextBisonLLM()
    text_llm_chain = LLMChain(prompt=text_summarizer_prompt_template,
                        llm=text_llm_obj) 
    picture_description = text_llm_chain.run(article=article)  
    picture_llm_obj = PictureGeneratorLLM()
    picture_llm_chain = LLMChain(prompt=picture_generator_prompt_template,
                                 llm=picture_llm_obj) 
    picture_url = picture_llm_chain.run(picture_prompt=picture_description) 
    return picture_url

if __name__ == "__main__": 
    import streamlit as st 
    st.title("My Story Visualizer")
    txt = st.text_area("Enter your paragraph", "") 
    if st.button("Submit", type="primary"): 
        picture_url = main_program(txt) 
        st.image(picture_url)