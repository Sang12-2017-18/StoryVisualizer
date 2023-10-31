from langchain.prompts import PromptTemplate

text_summarizer_prompt_template = PromptTemplate.from_template(
    """You are a picture description generator. Based on the article given below, give a description of a picture that would best describe the situation.
        Example: ARTICLE: I went to school in a fairly big bicycle. The school was so big, and all my teachers were nice.
        Answer: A cartoon picture of a teacher teaching her students in a classroom. Article: {article}"""
)

picture_generator_prompt_template = PromptTemplate.from_template(
    "{picture_prompt}"
)