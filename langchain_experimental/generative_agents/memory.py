import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseMemory, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.utils import mock_now

logger = logging.getLogger(__name__)


class GenerativeAgentMemory(BaseMemory):
    """Memory for the generative agent."""

    llm: BaseLanguageModel
    """The core language model."""
    memory_retriever: TimeWeightedVectorStoreRetriever
    """The retriever to fetch related memories."""
    social_media_memory: TimeWeightedVectorStoreRetriever

    product_memory: TimeWeightedVectorStoreRetriever
    verbose: bool = False
    ### Changing reflection threshold to be 100 so agent stops and reflects after aggregate score of 120
    reflection_threshold: Optional[float] = 500
    """When aggregate_importance exceeds reflection_threshold, stop to reflect."""
    current_plan: List[str] = []
    """The current plan of the agent."""
    # A weight of 0.15 makes this less important than it
    # would be otherwise, relative to salience and time
    importance_weight: float = 0.3
    """How much weight to assign the memory importance."""
    aggregate_importance: float = 0.0  # : :meta private:
    """Track the sum of the 'importance' of recent memories.
    Triggers reflection when it reaches reflection_threshold."""
    personalitylist: dict= {"extraversion": 0.0,"agreeableness":0.0, "openness":0.0, "conscientiousness":0.0,"neuroticism":0.0 }

    max_tokens_limit: int = 1500  # : :meta private:
    # input keys
    queries_key: str = "queries"
    most_recent_memories_token_key: str = "recent_memories_token"
    add_memory_key: str = "add_memory"
    # output keys
    relevant_memories_key: str = "relevant_memories"
    relevant_memories_simple_key: str = "relevant_memories_simple"
    most_recent_memories_key: str = "most_recent_memories"
    relevant_media_memories_key="relevant_media"
    now_key: str = "now"
    personality_key:str="personality"
    reflecting: bool = False

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        lines = [line for line in lines if line.strip()]  # remove empty lines
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def _get_topics_of_reflection(self, last_k: int = 25) -> List[str]:
        """Return the 2 most salient high-level questions about recent observations."""
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            "Given only the information above, what are the 2 most salient "
            "high-level questions we can answer about the subjects in the statements?\n"
            "Provide each question on a new line."
        )
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join(
            [self._format_memory_detail(o) for o in observations]
        )
        result = self.chain(prompt).run(observations=observation_str)
        return self._parse_list(result)
    

    def  search_prodct_questions(self, product, last_k: int = 25) -> List[str]:
        prompt = PromptTemplate.from_template(
            " Relevant Memories: {observations}\n\n"
            "Given these relevant information from a person's memories, what are seven relevant things you think they would search up to learn more about {product} \n"
            "Infer things to search up even if the given if the relevant information is not relevant to {product}. Make sure the questions relate to {product} and are specific questions"
            "Seperate each thing you want to learn with ;."
        )
        observations = self.fetch_socialmedia_memories(product)
        observation_str = "\n".join(
            [self._format_memory_detail(o) for o in observations]
        )
        result = self.chain(prompt).run(observations=observation_str,product=product)
        result= self._parse_list(result)
        result.pop(0)
        return result

    def _get_insights_on_topic(
        self, topic: str, now: Optional[datetime] = None
    ) -> List[str]:
        """Generate 'insights' on a topic of reflection, based on pertinent memories."""
        prompt = PromptTemplate.from_template(
            "Statements relevant to: '{topic}'\n"
            "---\n"
            "{related_statements}\n"
            "---\n"
            "What 5 high-level novel insights can you infer from the above statements "
            "that are relevant for answering the following question?\n"
            "Do not include any insights that are not relevant to the question.\n"
            "Do not repeat any insights that have already been made.\n\n"
            "Question: {topic}\n\n"
            "(example format: insight (because of 1, 5, 3))\n"
        )


        related_memories = self.fetch_memories(topic, now=now)
        related_statements = "\n".join(
            [
                self._format_memory_detail(memory, prefix=f"{i+1}. ")
                for i, memory in enumerate(related_memories)
            ]
        )
        result = self.chain(prompt).run(
            topic=topic, related_statements=related_statements
        )
        # TODO: Parse the connections between memories and insights
        return self._parse_list(result)
    
    def get_insights_recentmemories( self, last_k:int= 20, now:Optional[datetime]=None):
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            "Given these recent memories, what are the 4 most salient "
            "high-level insights we can make and infer from the above statements?\n"
            "Provide each insight on a new line."
        )
        observations = self.memory_retriever.memory_stream[-last_k:]
        observations += self.social_media_memory.memory_stream[-last_k:]
        observation_str = "\n".join(
            [self._format_memory_detail(o) for o in observations]
        )
        result = self.chain(prompt).run(observations=observation_str)
        return self._parse_list(result)

    def pause_to_reflect(self, now: Optional[datetime] = None) -> List[str]:
        """Reflect on recent observations and generate 'insights' regarding information about yourself."""
        print("Taking time reflecting on the current memories inputed")
        if self.verbose:
            logger.info("Character is reflecting")
        new_insights = []
        # topics = self._get_topics_of_reflection()
        # for topic in topics:
        insights= self.get_insights_recentmemories()
        for insight in insights:
            self.add_memory(insight, now=now)
            new_insights.extend(insights)
        self.updatepersonality()
        return new_insights

    def _score_memory_importance(self, memory_content: str) -> float:
        """Score the absolute importance of the given memory."""
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Respond with a single integer."
            + "\nMemory: {memory_content}"
            + "\nRating: "
        )
        score = self.chain(prompt).run(memory_content=memory_content).strip()
        if self.verbose:
            logger.info(f"Importance score: {score}")
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(match.group(1)) / 10) * self.importance_weight
        else:
            return 0.0
        
    ### Added function that scored the importance of a social media post based on 
    ### its length 
        
    def _score_mediapost_importance(self,memory_content:str)->float: 
        post_len= len(memory_content)
        score= post_len/12 
        if(score>10): return 10 
        else : 
            return int(score )

    def _score_memories_importance(self, memory_content: str) -> List[float]:
        """Score the absolute importance of the given memory."""
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Always answer with only a list of numbers."
            + " If just given one memory still respond in a list."
            + " Memories are separated by semi colans (;)"
            + "\Memories: {memory_content}"
            + "\nRating: "
        )
        scores = self.chain(prompt).run(memory_content=memory_content).strip()

        if self.verbose:
            logger.info(f"Importance scores: {scores}")

        # Split into list of strings and convert to floats
        scores_list = [float(x) for x in scores.split(";")]

        return scores_list
    #### Added function to add social media memory to the memory retriever 
    def add_socialmedia_memory(
        self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        print("adding memory")
        importance_score = self._score_mediapost_importance(memory_content)
        self.aggregate_importance += importance_score
        document = Document(
            page_content=memory_content, metadata={"importance": importance_score}
        )
        result = self.social_media_memory.add_documents([document], current_time=now)

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result
    def add_product_memory(self, memory_content:str, now:Optional[datetime]=None) -> List[str]: 
        "Add article to produc_memory "
        print("Adding product")
        document = Document(
            page_content=memory_content, metadata={"importance": .15}
        )
        result= self.product_memory.add_documents([document],current_time=now)
        return result
    

        

##### END of method 
    # def _get_topics_of_reflection(self, last_k: int = 25) -> List[str]:
    #     """Return the 3 most salient high-level questions about recent observations."""
    #     prompt = PromptTemplate.from_template(
    #         "{observations}\n\n"
    #         "Given only the information above, what are the 3 most salient "
    #         "high-level questions we can answer about the subjects in the statements?\n"
    #         "Provide each question on a new line."
    #     )
    #     observations = self.memory_retriever.memory_stream[-last_k:]
    #     observation_str = "\n".join(
    #         [self._format_memory_detail(o) for o in observations]
    #     )
    #     result = self.chain(prompt).run(observations=observation_str)
    #     return self._parse_list(result)


        # personalitylist= {"extraversion": 0.0,"agreeableness":0.0, "openness":0.0, "conscientiousness":0.0,"neuroticism":0.0 }

    def updatepersonality(self, now:Optional[datetime]=None, last_k: int =30):
            """"Update users personality traits based on recent memory stream"""
            prompt=PromptTemplate.from_template(
                "{memories}\n\n"
                "Given the following information about the person return a list where you rate"
                "the person's extraversion,agreeableness,openness to experience,conscientiousness, neuroticism  in that order"
                "on a scale of -1 to 1 where 1 means they fully posses the trait and -1 means they do posses the trait at all \n"
                "Seperate the list with semicolons. Here is an example output format .8 ; -.34; .23; -.54; .65")
            memories=self.social_media_memory.memory_stream[-last_k:]
            memories_str="\n".join(
            [self._format_memory_detail(o) for o in memories])
            result= self.chain(prompt).run(memories=memories_str)
            result_list= result.split(";")
            print("Personality Here")
            if(len(result_list)!=4) :
            #     return "Error"
            # for index, val in enumerate(result_list): 
            #     val.strip()
            #     self.personalitylist[index]+=float(val)

                self.personalitylist["extraversion"]+=float(result_list[0])
                self.personalitylist["agreeableness"]+=float(result_list[1])
                self.personalitylist["openness"]+=float(result_list[2])
                self.personalitylist["conscientiousness"]+=float(result_list[3])
                self.personalitylist["neuroticism"]+=float(result_list[4])
            return result 
            # return self._parse_list(result)
        


    def add_memories(
        self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observations or memories to the agent's memory."""
        importance_scores = self._score_memories_importance(memory_content)

        self.aggregate_importance += max(importance_scores)
        memory_list = memory_content.split(";")
        documents = []

        for i in range(len(memory_list)):
            documents.append(
                Document(
                    page_content=memory_list[i],
                    metadata={"importance": importance_scores[i]},
                )
            )

        result = self.memory_retriever.add_documents(documents, current_time=now)

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result

    def add_memory(
        self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        importance_score = self._score_memory_importance(memory_content)
        self.aggregate_importance += importance_score
        document = Document(
            page_content=memory_content, metadata={"importance": importance_score}
        )
        result = self.memory_retriever.add_documents([document], current_time=now)

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result

    def fetch_memories(
        self, observation: str, now: Optional[datetime] = None
    ) -> List[Document]:
        """Fetch related memories."""
        if now is not None:
            with mock_now(now):
                return self.memory_retriever.get_relevant_documents(observation)
        else:
            return self.memory_retriever.get_relevant_documents(observation)
    
    def fetch_socialmedia_memories(
        self, observation: str, now: Optional[datetime] = None
    ) -> List[Document]:
        """Fetch related memories."""
        # if now is not None:
        #     with mock_now(now):
        #         return self.memory_retriever.get_relevant_documents(observation)
        
        return self.social_media_memory.get_relevant_documents(observation)

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content = []
        for mem in relevant_memories:
            content.append(self._format_memory_detail(mem, prefix="- "))
        return "\n".join([f"{mem}" for mem in content])

    def _format_memory_detail(self, memory: Document, prefix: str = "") -> str:
        created_time = memory.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
        return f"{prefix}[{created_time}] {memory.page_content.strip()}"

    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

    def _get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
        result = []
        for doc in self.memory_retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc)
        return self.format_memories_simple(result)

    @property
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""
        return []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return key-value pairs given the text input to the chain."""
        queries = inputs.get(self.queries_key)
        now = inputs.get(self.now_key)
        if queries is not None:
            relevant_memories = [
                mem for query in queries for mem in self.fetch_memories(query, now=now)
            ]
            relevant_media_memories=[
                mem1 for query in queries for mem1 in self.fetch_socialmedia_memories(query, now=now)
            ]
            return {
                # self.personality_key: self.personalitylist,
                self.relevant_memories_key: self.format_memories_detail(
                    relevant_memories
                ),
                self.relevant_memories_simple_key: self.format_memories_simple(
                    relevant_memories
                ),
                self.relevant_media_memories_key: self.format_memories_detail(
                    relevant_media_memories
                )
            }

        most_recent_memories_token = inputs.get(self.most_recent_memories_token_key)
        if most_recent_memories_token is not None:
            return {
                self.most_recent_memories_key: self._get_memories_until_limit(
                    most_recent_memories_token
                )
            }
        return {}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the context of this model run to memory."""
        # TODO: fix the save memory key
        mem = outputs.get(self.add_memory_key)
        now = outputs.get(self.now_key)
        if mem:
            self.add_memory(mem, now=now)

    def clear(self) -> None:
        """Clear memory contents."""
        # TODO
