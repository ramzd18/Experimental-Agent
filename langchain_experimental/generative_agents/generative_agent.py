import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
import memory 
# from langchain_experimental.generative_agents.memory import GenerativeAgentMemory
from langchain_experimental.pydantic_v1 import BaseModel, Field


class GenerativeAgent(BaseModel):
    """An Agent as a character with memory and innate characteristics."""

    name: str
    """The character's name."""
    age: Optional[int] = None
    """The optional age of the character."""
    # traits: dict = {"extraversion": 0.0,"agreeableness":0.0, "openness":0.0, "conscientiousness":0.0,"neuroticism":0.0 }
    """Traits to ascribe to the character."""
    status: str
    """Current status of the agent"""
    education_and_work:str 
    """Agents current education_and_work"""
    interests: str
    """What the agent is interested in"""
    memory: memory.GenerativeAgentMemory
    """The memory object that combines relevance, recency, and 'importance'."""
    llm: BaseLanguageModel
    """The underlying language model."""
    verbose: bool = False
    summary: str = ""  #: :meta private:
    """Stateful self-summary generated via reflection on the character's memory.""" 
    summary_refresh_seconds: int = 3600  #: :meta private:
    """How frequently to re-generate the summary."""
    last_refreshed: datetime = Field(default_factory=datetime.now)  # : :meta private:
    """The last time the character's summary was regenerated."""
    daily_summaries: List[str] = Field(default_factory=list)  # : :meta private:
    """Summary of the events in the plan that the agent took."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    # LLM-related methods
    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory
        )

    def _get_entity_from_observation(self, observation: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}"
            + "\nEntity="
        )
        return self.chain(prompt).run(observation=observation).strip()

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}"
            + "\nThe {entity} is"
        )
        return (
            self.chain(prompt).run(entity=entity_name, observation=observation).strip()
        )        

    def summarize_related_memories(self, observation: str) -> str:
        """Summarize memories that are most relevant to an observation."""
        prompt = PromptTemplate.from_template(
            """
Summarize the following social media posts and memories relating to {observation}. If there are not relevant posts or information infer and make infromation that only relates to  {observation}. 
Include only information about {observation} in the summary. Do not include miscellanous information about the person and thier background. When you return the memories make sure to include that this is what {name} thinks.
Relevant social media posts:
{media}
Context from memory:
{memories}
Relevant context: 
"""
        )
        # entity_name = self._get_entity_from_observation(observation)
        # entity_action = self._get_entity_action(observation, entity_name)
        # q1 = f"What is the relationship between {self.name} and {entity_name}"
        # q2 = f"{entity_name} is {entity_action}"
        # q1=q1, queries=[q1, q2]
        relevmedia= self.memory.fetch_socialmedia_memories(observation)
        relevmemories = self.memory.fetch_memories(observation)
        return self.chain(prompt=prompt).run(observation=observation,media=relevmedia,memories=relevmemories,name=self.name).strip()
    
    ### Takes in personality trait and its corresponding integer value and return how disposed the agent is to that trait
    def definePersonalityValue(self, personality,num)-> str: 
        if(num<-.8): 
           return  f"{self.name} is extremely not {personality}"
        elif (num<-.3): 
            return f"{self.name} is not {personality}"
        elif (num<.1):
            return f"{self.name} is neutral on this personality trait {personality}"
        elif (num<.6):
            return f"{self.name} is {personality}"
        else: 
            return f"{self.name} is very {personality}"
        

    ## Takes in a pseronality dict with the 5 major personality traits and returns a string representing the agents overall personality 
    def definePersonality(self)-> str: 
        intitialstr=f"These are the {self.name} main personality traits" 
        for key in self.memory.personalitylist: 
            intitialstr+= " "+ self.definePersonalityValue(key, self.memory.personalitylist[key]) +"\n"
        return intitialstr



    def _generate_reaction(
        self, observation: str, suffix: str, now: Optional[datetime] = None
    ) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            # + "\n{agent_name}'s status: {agent_status}"
            +"\n Agents's major personalities traits:"
            +"\n{personality}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            # + "\nMost recent observations: {most_recent_memories}"
            +"\n Use the following information to generate an agent's response to the following observation"
            + "\nObservation: {observation}"
            + "\n\n"
            + suffix
        )
        agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            observation=observation,
            personality=self.definePersonality(),
            # agent_status=self.status,
        )
        # consumed_tokens = self.llm.get_num_tokens(
        #     prompt.format(most_recent_memories="", **kwargs)
        # )
        # kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        return self.chain(prompt=prompt).run(**kwargs).strip()

    def _clean_response(self, text: str) -> str:
        return re.sub(f"^{self.name} ", "", text.strip()).strip()

    # def generate_reaction(
    #     self, observation: str, now: Optional[datetime] = None
    # ) -> Tuple[bool, str]:
    #     """React to a given observation."""
    #     call_to_action_template = (
    #         "Should {agent_name} react to the observation, and if so,"
    #         + " what would be an appropriate reaction? Respond in one line."
    #         + ' If the action is to engage in dialogue, write:\nSAY: "what to say"'
    #         + "\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
    #         + "\nEither do nothing, react, or say something but not both.\n\n"
    #     )
    #     full_result = self._generate_reaction(
    #         observation, call_to_action_template, now=now
    #     )
    #     result = full_result.strip().split("\n")[0]
    #     # AAA
    #     self.memory.save_context(
    #         {},
    #         {
    #             self.memory.add_memory_key: f"{self.name} observed "
    #             f"{observation} and reacted by {result}",
    #             self.memory.now_key: now,
    #         },
    #     )
    #     if "REACT:" in result:
    #         reaction = self._clean_response(result.split("REACT:")[-1])
    #         return False, f"{self.name} {reaction}"
    #     if "SAY:" in result:
    #         said_value = self._clean_response(result.split("SAY:")[-1])
    #         return True, f"{self.name} said {said_value}"
    #     else:
    #         return False, result

    # def generate_dialogue_response(
    #     self, observation: str, now: Optional[datetime] = None
    # ) -> Tuple[bool, str]:
    #     """React to a given observation."""
    #     call_to_action_template = (
    #         "What would {agent_name} say? To end the conversation, write:"
    #         ' GOODBYE: "what to say". Otherwise to continue the conversation,'
    #         ' write: SAY: "what to say next"\n\n'
    #     )
    #     full_result = self._generate_reaction(
    #         observation, call_to_action_template, now=now
    #     )
    #     result = full_result.strip().split("\n")[0]
    #     if "GOODBYE:" in result:
    #         farewell = self._clean_response(result.split("GOODBYE:")[-1])
    #         self.memory.save_context(
    #             {},
    #             {
    #                 self.memory.add_memory_key: f"{self.name} observed "
    #                 f"{observation} and said {farewell}",
    #                 self.memory.now_key: now,
    #             },
    #         )
    #         return False, f"{self.name} said {farewell}"
    #     if "SAY:" in result:
    #         response_text = self._clean_response(result.split("SAY:")[-1])
    #         self.memory.save_context(
    #             {},
    #             {
    #                 self.memory.add_memory_key: f"{self.name} observed "
    #                 f"{observation} and said {response_text}",
    #                 self.memory.now_key: now,
    #             },
    #         )
    #         return True, f"{self.name} said {response_text}"
    #     else:
    #         return False, result
        
    def generate_question_response(self, question:str, now: Optional[datetime]=None)->str: 
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            # + "\n{agent_name}'s status: {agent_status}"
            +"\n Agents's major personalities traits:"
            +"\n{personality}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            # + "\nMost recent observations: {most_recent_memories}"
            +"\n Use the following information to generate a response from {agent_name}'s perspective. Anwser the question and only anwser from {agent_name}'s perspective. Do not include anything about being an AI model. "
            + "\nThe questions being asked: {question}"
            + "\n\n"
            # + suffix
        )
    
        agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = self.summarize_related_memories(question)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=self.education_and_work+ "    "+ self.interests,
            relevant_memories=relevant_memories_str,
            personality=self.memory.personalitylist,
            agent_name= self.name,
            question=question

            # agent_status=self.status,
        )
        
        return self.chain(prompt).run(**kwargs).strip()
    
    # def daily_scheudle_memoreies(self, now:Optional[datetime]=None)-> str: 
    #     prompt=PromptTemplate.from_template(
    #        " {agent_summary_description}"

    #         # + "\nMost recent observations: {most_recent_memories}"
    #         +"\n Use the above information about a person and their social media posts put below to generate a realistic daily schedule of what the person does everyday, what tasks they use, what products they use, who they talk to"
    #         + "\nSocial Media History:{socialmedia}"
    #         + "\n\n"

    #     )
    #     return "test"


    ### Generic function that takes in a prompot and general observation and outputs a list of memories to be added to the agents real world memory 
    def generic_social_media_addmemories(self,topic:str,  promptstr:str, now:Optional[datetime]=None)-> str: 
         prompt=PromptTemplate.from_template(
           " Summmary of person:  {agent_summary_description}"
            + "\nSocial Media History:{socialmedia}"
            + "\n{promptstr}" 
        )
         
         agentsummary= self.get_summary(now=now)
         socialmediastr= self.memory.fetch_socialmedia_memories(topic)
         kwargs: Dict[str, Any] = dict(
            agent_summary_description=agentsummary,
            socialmedia=socialmediastr,
            promptstr=promptstr,
        )
         response= self.chain(prompt=prompt).run(**kwargs).strip()
         memories= response.split(";")
         for mem in memories: 
             self.memory.add_memories(mem)

         return memories
    def product_to_memory(self, prodcut):
        total=len(self.memory.product_memory.vectorstore.index_to_docstore_id)
        iter=total/6
        
        lowerbounds=0
        upperbounds=lowerbounds+iter
        if(total<6 ):
            upperbounds=total-1
            iter=2

        
        documentstr=""; 
        social_mem=self.memory.fetch_socialmedia_memories(prodcut)
        while upperbounds<total:
            print("product memorying")
            memorystream= self.memory.product_memory.memory_stream[lowerbounds:upperbounds]
            observation_str = "\n".join(
            [self.memory._format_memory_detail(o) for o in memorystream
             ])
            prompt = PromptTemplate.from_template(
            "Here is a list of articles about {product}: "
            "---\n"
            "{observation_str}\n"
            "---\n"
            "Here are a list of {name}'s relevent memories towards {product}  on social media:"
            "---\n"
            "{social_str}\n"
            "---\n"
            "Here is a summmary of Ram: {summary} \n"
            "Here are Ram's personality traits: {personality} \n"
            "Given this generate an extensive list of insights {name} would have based on reading these articles. Write the insights from the perspective of {name} and only include {name}'s personal insights and how they relate to his information. Seperate them with a ;"
        )
            result =self.memory.chain(prompt).run(product=prodcut,observation_str=observation_str,name=self.name,social_str=social_mem,summary=self.education_and_work+" Interests: "+self.interests,personality=self.memory.personalitylist)
            self.memory.add_memories(result)
            print(result)
            lowerbounds=upperbounds
            upperbounds=upperbounds+iter


    ######################################################
    # Agent stateful' summary methods.                   #
    # Each dialog or response prompt includes a header   #
    # summarizing the agent's self-description. This is  #
    # updated periodically through probing its memories  #
    ######################################################
    def set_education(self,edu): 
        self.education_and_work=edu


    def _compute_agent_summary(self) -> str:
        """"""
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s as a person given the"
            + " following statements:\n"
            + "{relevant_memories}"
            + "Do not embellish."
            + "\n\nSummary: "
        )
        # The agent seeks to think about their core characteristics.
        return (
            self.chain(prompt)
            .run(name=self.name, queries=[f"{self.name}'s core traits"])
            .strip()
        )

    def get_summary(
        self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:
        """Return a descriptive summary of the agent."""
        current_time = datetime.now() if now is None else now
        since_refresh = (current_time - self.last_refreshed).seconds
        if (
            not self.summary
            or since_refresh >= self.summary_refresh_seconds
            or force_refresh
        ):
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time
        age = self.age if self.age is not None else "N/A"
        return (
            f"Name: {self.name}"
            +f"\n Age:{age}"
            +f"\n {self.name} educational history and work history {self.education_and_work}"
            +f"\n {self.name} personality traits {self.definePersonality()}"
            # + f"\n An initial small summary {self.summary}"
        )

    # def get_full_header(
    #     self, force_refresh: bool = False, now: Optional[datetime] = None
    # ) -> str:
    #     """Return a full header of the agent's status, summary, and current time."""
    #     now = datetime.now() if now is None else now
    #     summary = self.get_summary(force_refresh=force_refresh, now=now)
    #     current_time_str = now.strftime("%B %d, %Y, %I:%M %p")
    #     return (
    #         f"{summary}\nIt is {current_time_str}.\n{self.name}'s status: {self.status}"
    #     )
