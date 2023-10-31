## Create a custom class in Langchain that creates LLM.
import google.generativeai as palm
from langchain.llms.base import LLM
from typing import Mapping, Optional, Any, List
from langchain.callbacks.manager import CallbackManagerForLLMRun 

from env import PALM_API_KEY, TEXT_SUMMARIZER_MODEL_NAME

class TextBisonLLM(LLM): 
  model_name = TEXT_SUMMARIZER_MODEL_NAME
  temperature = 0.5 
  max_output_tokens = 200 

  @property
  def _identifying_params(self) -> Mapping[str, Any]: 
    params_dict = {
      "model_name": self.model_name,
      "temperature": self.temperature,
      "max_output_tokens": self.max_output_tokens, 
    }
    return params_dict

  @property
  def _llm_type(self) -> str:
    return "custom"

  def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
    palm.configure(api_key=PALM_API_KEY)
    completion = palm.generate_text(
          model=TEXT_SUMMARIZER_MODEL_NAME,
          prompt=prompt,
          temperature=self.temperature,
          # The maximum length of the response
          max_output_tokens=self.max_output_tokens,
      )
    return completion.result