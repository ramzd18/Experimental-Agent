import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel

from langchain_experimental.generative_agents.memory import GenerativeAgentMemory
from langchain_experimental.pydantic_v1 import BaseModel, Field
import queue
import threading
import time
from openai import OpenAI
import base64
import requests
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
    memory: GenerativeAgentMemory
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
    def chain1(self, prompt: PromptTemplate) -> LLMChain:
        llm1 = ChatOpenAI(model_name='gpt-4',temperature=0.35)
        return LLMChain(
            llm=llm1, prompt=prompt, verbose=self.verbose, memory=self.memory
        )
    def chain2(self, prompt: PromptTemplate) -> LLMChain:
        llm1 = ChatOpenAI(model_name='gpt-4',temperature=0.8)
        return LLMChain(
            llm=llm1, prompt=prompt, verbose=self.verbose, memory=self.memory
        )
    def chain3(self, prompt: PromptTemplate) -> LLMChain:
        llm1 = ChatOpenAI(model_name='gpt-4-1106-preview',temperature=0.6)
        return LLMChain(
            llm=llm1, prompt=prompt, verbose=self.verbose, memory=self.memory
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
A person is being interviewed for user research by a company. They ask about {observation}. Use the following social media posts and memories from this person to crete information and opinions the person has about the question. If there are not relevant posts or information infer and make infromation that only relates to  {observation}. 
\ Do not include miscellanous information about the person and thier background. When you return the memories make sure to include that this is what {name} thinks. Make it detailed and infer information to make it as detailed as possible.
Make sure you are using the following posts and memories to make the information.
Relevant social media posts:
{media}
Context from memory:
{memories}
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
        # relevant_memories_str = self.summarize_related_memories(observation)
        relv_memories= self.memory.fetch_memories(observation)
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
    
    def  search_prodct_questions(self, product,status,  last_k: int = 25) -> List[str]:
        prompt = PromptTemplate.from_template(
            "Summary of {name}: "
            "{summary}"
            " Relevant Memories: {observations}\n"
            "Given theis relevant information from a persons social media, what are three relevant things you think they would search up to learn more about {product} \n"
            "Infer things to search up even if the given if the relevant information is not relevant to {product}. Make sure the questions relate to {product} and are specific questions about {product}"
            "Seperate each thing you want to learn with ;."
        )
        observations = self.memory.fetch_socialmedia_memories(product)
        print("searching product stuff up")
        summary=self.get_summary()
        observation_str = "\n".join(
            [self.memory._format_memory_detail(o) for o in observations]
        )
        result = self.chain(prompt).run(name=self.name,observations=observation_str,summary=summary, product=product)
        result= result.split(";")
        return result
    def  search_description_questions(self, description,  last_k: int = 25) -> List[str]:
        prompt = PromptTemplate.from_template(
            "You are being tasked with generating questiosn a person would search online."
            "Summary of {name}: "
            "{summary}"
            " Here ia a description of the person that was inputted: {description}"
            "This description likely details a problem or general persona a person fits under."
            "Given this relevant information about the person from their summary, what are twelve relevant things you think they would search up to fit the desciprtion that was inputted. \n"
            "Tailor the questions so you are covering what products this persons may use to solve their problems/fit their persona.For example if the person inputted description of a person struggling to mantain personal finances, you might look up different budgeting products online, their pros/and cons, the basics of maintaining budget, and other relevant things to help you be more knowledagble about the topic."
            " Make sure the questions relate to {description}."
            "MAKE SURE YOU GENERATE 12 QUESTIONS. DO NOT RETURN LESS QUESTIONS. MAKE 12. SEPERATE EACH ONE OF THE QUESTIONS WITH A ; when you return them."
        )
        summary=self.get_summary()
        result = self.chain(prompt).run(name=self.name,summary=summary,description=description)
        result= result.split(";")
        print(result)
        return result

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
            "Imagine you are a {agent_name} and you are being interviewed by a company to better understand you and your perspective. Here is some information about you:\n"
            "{agent_summary_description}"
            + "\n{agent_name}'s status: {agent_status}"
            +"This status represents the general description inputted for you by the company."
            +"\n {agent_name}'s interests:"
            +"\n{interests}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            # + "\nMost recent observations: {most_recent_memories}"
            + "This is the question you are being asked {question}"
            +" Anwser the following question from {agent_name} perspective. Make sure the response is personalized to you and not something you would imagine everyone says. Make it unique to you. Only include relevant information that anwsers the question and make the response concise with only information that directly anwsers the question."
            +"Make the response as human like as possible and personable. Anwser directly as {agent_name} and use personal pronouns"
            +"Be consice and do not include irrelevant question. Anwser the question directly and that is it. Do not give a long-winded response that is not related."
            + "\n\n"
            # + suffix
        )
        interests=str(self.interests)
        agent_summary_description = self.get_summary(now=now)
        # relevant_memories_str = self.summarize_related_memories(question)
        memstr=""
        relvmems= self.memory.fetch_memories(question)
        for doc in relvmems: 
            memstr+="New Memory: "+ doc.page_content
        # print(relevant_memories_str)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=self.education_and_work+ "    "+ self.interests,
            relevant_memories=memstr,
            interests=self.interests,
            agent_name= self.name,
            question=question,
            agent_status=self.status

            # agent_status=self.status,
        )
        result= self.chain1(prompt).run(**kwargs).strip()
        self.memory.add_memory(result)
        return result
    def generate_question_response_interview(self, question:str, interview_context, now: Optional[datetime]=None)->str: 
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "Imagine you are a {agent_name} and you are being interviewed by a company to better understand you and your perspective. Here is some information about you:\n"
            "{agent_summary_description}"
            + "\n{agent_name}'s status: {agent_status}"
            +"This status represents the general description inputted for you by the company."
            +"\n {agent_name}'s interests:"
            +"\n{interests}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            # + "\nMost recent observations: {most_recent_memories}"
            +f"Here is the past questions and anwsers you have been asked during the interviw: {interview_context}"
            + "This is the question you are being asked {question}"
            +" Anwser the following question from {agent_name} perspective. Make sure the response is personalized to you and not something you would imagine everyone says. Make it unique to you. Only include relevant information that anwsers the question and make the response concise with only information that directly anwsers the question."
            +"Make the response as human like as possible and personable. Anwser directly as {agent_name} and use personal pronouns. Make sure your anwser is consistent with your responses previosuly in the interview but do not repeat information."
            +"Do not just have a positive response when asked questions. You can anwser NO and say you do not like something or a product when asked about your opinion. Make your opinion known. If you would not like a product the company is asking about you can say that."
            +"Make sure the response is unique to you and something you believe you would say. Do not just give the most common anwser but anwser what makes sense to you and your profile. Incorporate your relevant memories and provide a unique response that represents you. DO NOT JUST GIVE A COMMON ANWSER YOU THINK EVERYONE WOULD SAY."
            +"DO NOT PROVIDE THE SAME ANWSER AS YOU WOULD WITH EVERY PROFILE. Respond with an anwser that fits this persons profile. I would not want you to generate the same anwser for multiple different people as that would defeat the purpose. Be unique and be creative."
            +"Be consice and do not include irrelevant questions. Anwser the question directly and that is it. Do not give a long-winded response that is not related."
            + "\n\n"
            # + suffix
        )
        interests=str(self.interests)
        agent_summary_description = self.get_summary(now=now)
        # relevant_memories_str = self.summarize_related_memories(question)
        memstr=""
        relvmems= self.memory.fetch_memories(question)
        for doc in relvmems: 
            memstr+="New Memory: "+ doc.page_content
        # print(relevant_memories_str)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=self.education_and_work+ "    "+ self.interests,
            relevant_memories=memstr,
            interests=self.interests,
            agent_name= self.name,
            question=question,
            agent_status=self.status

            # agent_status=self.status,
        )
        result= self.chain2(prompt).run(**kwargs).strip()
        self.memory.add_memory(result)
        return result
    
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
    

    def turn_soc_memories_into_list(self,mems,now: Optional[datetime]=None):
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            # + "\n{agent_name}'s status: {agent_status}"
            +"\n {agent_name}'s interests:"
            + "\nAll of  {agent_name}'s current memories:"
            + "\n{relevant_memories}"
            + "\n\n"
            +"\n Use the following information to generate a response a list of specific things {agent_name} would search up to learn more about on the internet."
            # + suffix
        )
    
        # agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = mems
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=self.education_and_work+ "    "+ self.interests,
            relevant_memories=relevant_memories_str,
            agent_name= self.name,
        )
        result= self.chain(prompt).run(**kwargs).strip()
        return result
    def memoriesprompt(self,skills,mems, now: Optional[datetime]=None):
        prompt = PromptTemplate.from_template(
            "I am trying to rebuild a persons memory. I have some information about them and what they posted on social media. Here is the information."
            +"name:"
            "{agent_summary_description}"
            # + "\n{agent_name}'s status: {agent_status}"
            +"\n {agent_name}'s skills:"
            +"{skills}"
            + "\nSome of  {agent_name}'s relevant social media posts:"
            + "\n{relevant_memories}"
            + "Here are the existing memories we have of {agent_name}"
            +"{memories}"
            + "\n"
            +"\n Use the following information to generate a list of 30 additional memories for the person to bolster the information we have on them. The memories {agent_name} already contains are limited and  I want you to produce more memories to make the information about {agent_name} more complete. Write them from perspective of {agent_name}. Use personal pronouns."
            +"Do not give basic memroies stating that you remember doing something. Give memories with extensive detail that will help build a more complete profile. For example a bad memory would be: I remember the first time I walked into school. A good memory woudl be: The first day of college I had to move in, but I was not prepared to set up my room and live with my unknown roomate."
            +"Create memories that are different than the existing memories. DO NOT repeat memories."
            +"Generate the memories with extensive detail.Do not use words like I remember and instead use words that create active and detailed memories about their life. Make sure to include detials like specific names, venues, or actions. They should be able to fill out {agent_name}'s life and make our information about them more relevant. Seperate each memory with a semicolon.Example format: mem1;mem2;mem3;mem4 ..."
            # + suffix
        )
    
        # agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = self.memory.fetch_socialmedia_memories(self.interests)
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=self.education_and_work+ "\n"+"Interests:" + self.interests,
            relevant_memories=relevant_memories_str,
            agent_name= self.name,
            skills=skills,
            memories=mems,
        )
        result= self.chain3(prompt).run(**kwargs)
        return result
    def memoriespromptkeyword(self,mems,keyword,now: Optional[datetime]=None):
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            # + "\n{agent_name}'s status: {agent_status}"
            +"\n {agent_name}'s interests:"
            + "\nAll of  {agent_name}'s current memories:"
            + "\n{relevant_memories}"
            + "\n\n"
            +"\n Use the following information to generate a comprehensive list of additional memories the person may have about {keyword} to bolster the information we have on them. The memories {agent_name} already contains are limited and  I want you to produce more memories to make the information about {agent_name} more complete. Generate as many memories as you can. Seperate each memory with a semicolon."
            # + suffix
        )
    
        # agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = mems
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=self.education_and_work+ "    "+ self.interests,
            relevant_memories=relevant_memories_str,
            agent_name= self.name,
            keyword=keyword,
        )
        result= self.chain(prompt).run(**kwargs).strip()
        return result
    def analysis_of_product(self,list_of_text,description):
        total_len= len(list_of_text)
        iter= total_len//8
        if (total_len<7): 
            iter=1
        begin=0
        end=iter
        totallist=[]
        while(end<total_len-1): 
            sublist= list_of_text[begin:end]
            resultque=queue.Queue()
            task_thread = threading.Thread(target=self.memoryfunc, args=(sublist,resultque,description,str(totallist)))
            task_thread.start()
            task_thread.join(timeout=20)
            if task_thread.is_alive():
               print("skip")
            else:
                print("did nto skip")
                result=resultque.get()
                for memory in result: 
                    print(memory)
                    # self.memory.add_memory(memory)
                    totallist.append(memory)
            begin=end
            print("Lowerboubds"+str(begin))
            end=end+iter
            print("Up"+str(end))
        return totallist
    def marketing_analysis(self,first,context):
        prompt=PromptTemplate.from_template(
            "You are a person who is being interviewed by a company to understand what mareting materials you like more. Here is information about you: \n"
            "Here is the context of the product you are evaluating:"
            "{context}"
            "Here is relevant information about yourself: "
             "{agent_summary_description}"
            # + "\n{agent_name}'s status: {agent_status}"
            "\n {agent_name}'s interests:"
            "\n{interests}"
             "\nSummary of relevant context from {agent_name}'s memory:"
             "\n{relevant_memories}"
             "Your status: {status}"
            "You are going to rate the marketing material."
            "For the material you are going to score it on clarity, personalization, Impact,and Retention Time.\n"
            "Clarity represents how clear the material was and how easily you understood it."
            "Personalization represents how well the material speaks to you and relates to you."
            "Impact represents how powerful the message is the power it had to call you to action."
            "Retention Time represents how long you wpend viewing or interacting with the material."
            "You are gonna rate each metric for the material on a scale of 0 to 1 and also return a new material that you think would sell more. Make sure the new optimized material you return be similair length to the inputted material. So if the input is 8 words your output shoult not be longer than 13 words. Return the output in a list. "
            "Make sure the new tagline is catchy and makes sense for the brand. Do not make too overly personalized to you. Make something you think many people would like"
            "Here is an example return format: [clarity: .42,personalization:.51,impact:.12,retention_time:.62, optimized_message: This is the better version of the mssage according to you ]. Make sure it follows this format and uses semicolons as shown in the example."
            "Here is the material: \n"
            "{first}"
           
        )
        interests=str(self.interests)
        agent_summary_description = self.get_summary()
        # relevant_memories_str = self.summarize_related_memories(question)
        memstr=""
        # question= first+ " "+second
        relvmems= self.memory.fetch_memories(first)
        for doc in relvmems: 
            memstr+="New Memory: "+ doc.page_content
        # print(relevant_memories_str)
        # current_time_str = (
        #     datetime.now().strftime("%B %d, %Y, %I:%M %p")
        #     if now is None
        #     else now.strftime("%B %d, %Y, %I:%M %p")
        # )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=self.education_and_work+ "    "+ self.interests,
            relevant_memories=memstr,
            interests=self.interests,
            agent_name= self.name,
            first=first,
            context=context,
            status=self.status
        )

        result= self.chain1(prompt).run(**kwargs)
        return result

    def memoryfunc(self,list,resultque,description,pastmems): 
            prompt = PromptTemplate.from_template(
            "I was given a description of a person. Here is the description: {description}"
            +"I am making an ai replica of this person and thier name is {name}"
           + "Here is a list of summarized articles {name} searched up on the internet relating to the description that was inputted of them. "
            "{observation_str}\n"
            "---\n"
            "This description likely depicts a specific thing or problem this person embodies and I am trying to generate relevant memories for this person regarding the description using what I already know about them. Here is the information I already know: "
            "Here are a summary of {name}'s relevent memories towards the topics of these articles:"
            "{social_str}\n"
            "{past_mems}"
            "---\n"
            "Here is a summary of {name}: {summary}"
            # "Here are {name}'s interests: {interests} \n"
            #  "{name}'s current status: {status} \n"
            ""
            " Imagine that you are {name}. Given this generate a list of memories {name} would remember based on reading these articles. Write the memories from the perspective of {name}. Make sure they are personalized memories. "
            "Write as many memories as you can. Seperate the memories with a semicolon.Do not repeat existing memories the person has had. You can build on existing ones but make sure they are distinct and each one helps builds a complete profile of this person for this topic."
            "MAKE SURE THESE MEMORIES ABOUT THE ARTICLE. Make it so these memories relate directly to the description that was inputted and recreate this person's memories about the topic. I am trying to recreate as many realistic memories about the topic of the description as possible."
            "For example, if the person we were reading an article about basketball shoes, and the person enjoyed playing basketball a memory coud be I played countless games of pick up basketball with my friends and tried jumping so hard my shoe broke."
            "Avoid using works like I remember or I recall or I am feeling. and instead state the memory directly and include extremely specific details in the memory so they are not broad or general. Be as creative as you can be. Let the memories be unique and use the articles and the persons profile to make the most realistic human like memories possible. Do not just reuse the information from the articles. Think about how they might have applied to you and your life and make sure you are unqiue to the person."
            "Here is an example format memory1;memory2;memory3;memory4;memory5;memory6 and so on"
        )
            soc_mem=self.summarize_related_memories(str(list))
            result =self.chain3(prompt).run(observation_str=str(list),name=self.name,social_str=soc_mem,summary=self.get_summary(),interests=str(self.interests),status=self.status,description=description,past_mems=pastmems)
            result=result.split(";")
            print(result) 
            resultque.put(result)

    def memorygenerate(self,description): 
            prompt = PromptTemplate.from_template(
            "I was given a description of a person. Here is the description: {description}"
            +"I am making an AI replica of this person and thier name is {name}"
        #    + "Here is a list of summarized articles {name} searched up on the internet relating to the description that was inputted of them. "
        #     "{observation_str}\n"
        #     "---\n"
            "This description likely depicts a specific thing or problem this person embodies and I am trying to generate relevant memories for this person regarding the description using what I already know about them. Here is the information I already know: "
            "Here are a summary of {name}'s relevent memories towards the description:"
            "{social_str}\n"
            "---\n"
            "Here is a summary of {name}: {summary}"
            # "Here are {name}'s interests: {interests} \n"
            #  "{name}'s current status: {status} \n"
            ""
            " Imagine that you are {name}. Given this generate 40 memories {name} would have regarding the description. Specifically make memories of them using proucts, facing problems, or solving issues related to the description. For example if the description was someone who enjoys wine you would generate extensive memories about what specific wine they like, what brands, what products they use, what potential issues they have had and so on. Write the memories from the perspective of {name}. Make sure they are personalized memories. "
            "For example, if someone had inputted a description about a person struggling to manage their personal finances you would generate apps about what apps they use to budget, what they like and do not like about these apps, their biggest issues budgeting, where they spend the most money, and any other experiences they might have with their personal finances. It should be to this level of detail and incorporate personal information used above."
            "Write as many memories as you can. Seperate the memories with a semicolon."
            "Make it so these memories relate directly to the description that was inputted and recreate this person's memories about the topic. I am trying to recreate as many realistic memories about the topic of the description as possible."
            "Avoid using works like I remember or I recall or I am feeling. and instead state the memory directly and include extremely specific details in the memory so they are not broad or general. Be as creative as you can be. Let the memories be unique and use the articles and the persons profile to make the most realistic human like memories possible. Do not just reuse the information from the articles. Think about how they might have applied to you and your life and make sure you are unqiue to the person."
            "Make the memories persoanl to the profile of the person. Do not just have general memories. Tailor it so it is realistic. This is important because I will make multiple people and I want to be distinct so it is important their memories are not the same."
            "Here is an example format memory1;memory2;memory3;memory4;memory5;memory6 and so on"
        )
            soc_mem=self.summarize_related_memories(str(list))
            result =self.chain1(prompt).run(name=self.name,social_str=soc_mem,summary=self.get_summary(),interests=str(self.interests),status=self.status,description=description)
            result=result.split(";")
            return result
    ## Function used when agent is initialized. Stores relevant memory about a specific product. 
    def product_to_memory(self, prodcut):
        total=len(self.memory.product_memory.vectorstore.index_to_docstore_id)
        print("Total Value is +"+str(total))
        iter=total//5
        
        lowerbounds=0
        upperbounds=lowerbounds+iter
        if(total<5):
            upperbounds=lowerbounds+1
            iter=2

        print("lowerbound"+str(lowerbounds)+"Upperbound"+str(upperbounds))
        documentstr=""; 
        social_mem=self.memory.fetch_socialmedia_memories(prodcut)
        print("Social Memmories: "+str(social_mem))
        while upperbounds<total:
            print("product memorying")
            memorystream= self.memory.product_memory.memory_stream[lowerbounds:upperbounds]
            observation_str = "\n".join(
            [self.memory._format_memory_detail(o) for o in memorystream
             ])
            
            print("observation string is this:"+observation_str)
            prompt = PromptTemplate.from_template(
            "Here is a list of articles about {product}: "
            "{observation_str}\n"
            "---\n"
            "Here are a list of {name}'s relevent memories towards {product}  on social media:"
            "{social_str}\n"
            "---\n"
            "Here are {name}'s interests: {interests} \n"
             "{name}'s current status: {status} \n"
            "Given this generate a list of insights {name} would have based on reading these articles. Write the insights from the perspective of {name} and only include {name}'s personal insights and how they relate to their information and current situation. Make sure they are personalized insights. Seperate the insights with a semicolon."
            "Here is an example format  insight1; insight2;insight3;insight4;insight5;insight6 and so on"
        )
            print("finished prompt")
            result =self.memory.chain(prompt).run(product=prodcut,observation_str=observation_str,name=self.name,social_str=social_mem,summary=self.get_summary(),interests=self.interests,status=self.status)
            result=result.split(";")
            print("finished chain"+ str(len(result)))
            for memory in result: 
                print("adding mem now")
                self.memory.add_memory(memory)
            lowerbounds=upperbounds
            print("Lowerboubds"+str(lowerbounds))
            upperbounds=upperbounds+iter
            print("Up"+str(upperbounds))

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
            +f"\n {self.name}'s status {self.status}"

            # + f"\n An initial small summary {self.summary}"
        )
    # def encode_image(image_path):
    #     with open(image_path, "rb") as image_file:
    #         return base64.b64encode(image_file.read()).decode('utf-8')




# savedprompt=           "text": f"Here is context of the website you are looking at.{website_context}.Here is a summary of yourself: {summary}."
#                  +"Use this summmary of yourself and specially your response about what you want from this website and what specifically you would want to use it for to guide your responses."
#                  +" Given these sets of pictures of the website respond with wheter you want to click something on the website or if you want to type something into an element. Respond with your anwser in a dictionary with either button or search as they key and object name you are clicking or search value as the value. If you want to click on image or text that you think is clickable provide the text or image caption as the clickable value in the dictionary."
#                  +"For example if you wanted to click a button named Submit you would return button: Submit. If you wanted to click a text labeled Click This you would return button: Click This. If you wanted to type 240 dollars into a box you would return search: 240 dollars. Use quotations. Only return this value and nothing else."
#                  +f"Sometimes it may be hard for you to identify if something is clickable. Here is a list of the clickable elements you can choose on the page to help you determine what is clickable: {clickable_elements}. You may not see some of these elements in the screensot. If you want to choose an element in the list and is not in the picture do not pick it. Only pick what you can see."
#                  +"Be aware of popups. You may need to click a specific button to get to the actual page. In this case chose the button or exit out of the popup."
#                 +f" Here is the past things you have searched/click on the website: {past_context}. If you  something was already clicked or search in the past thigns you have searched/click do not return it again. DO NOT REPEAT things. For example if you clicked a buttonn called Best Clothes Here, do not pick that button again if you have choose it recently. This will cause issues by causing you to go in circles."
#                 +"PICK UNIQUE ELEMENTS THAT ARE NOT FOUND IN YOUR PAST THINGS UNLESS NECESSARY. DO not be afraid to search things up do not only rely on button quicks. You should be emulating hwo someone is actually using this platform so make it realistic."
#                   +" Make it realistic to how your profile might use the website. In the dictionary you are returning I also want you to return one other value. Add the key feedback in the dictionary and as a value put what you thought of the pages you are seeing. Wheter you like what you see, if they are intuitive and what could be changed about the page to make it better for you. So for example you might have feedback: The information presented does not line up with what I thought. I expected that all the posts would be centralized but I have to click to get posts about a specific topic which takes a lot of time.Make the feedback about what you see currently and what information is being shwon to you. Be specific and tailor to it what you would specifically want changed."
#                   +"For the feedback be aware that you might not click or see features immediately so try to only give feedback on what can you see. THink about why you might use this product and what you want out of it and how it is living up to your expectations. Highlight anything that is potentially confusing or diffucult to understand. Avoid general feedback like the design is ver intuivtive and easy to understand. Go into specific details and highlight how specific parts are affecting your user experience. Be specific and highlight your journey."
#                   +"For the feedback be aware of popups. If you see a popup only comment on the popup and not the main content. When you provide feedback make sure you look at everything in the image. Do not say you wish there was a feature if it is present in the main content becaause you may have not seen it. Think about what you would like the most on your user journey and what specifically you think would improve your experience and make the software better."
#                   +"Make the feedback important imapctful information that would help people improve the website you are looking at. Go beyond basic feedback. Talk about deep details, quality of information, usabillity and thinks of this nature that will help improve the website. Think about potential replacements to your criticism and how it would effect your experience, highlight this."
#                   +"Add this to dictionary you are returning."


#     example_prompt=f"""You are an AI agent testing a website as a specific user persona. Your task is to interact with the website and provide detailed feedback.

# Website Context
# Context: {website_context}
# \n
# Your Persona
# Here is a Summary of yourself including what you want out of the website: {summary}
# \n
# Task Instructions:
# You are going to either choose if you want to click an element or search something up based on the elments in the screenshot.
# Interaction:
# Clickable Elements: Here is a list of all clickable elements: {clickable_elements}, If you want to click something choose something from here.
# Action Format:
# For clicks, return button: [item name]. Example: button: Submit
# For typing, return search: [search value]. Example: search: 240 dollars
# Note: Only interact with elements visible in the screenshot.
# Past Interactions:
# Avoid repeating past actions. Here is a list of what you have previosly clicked: {past_context} for history.
# Focus on unique elements unless repetition is necessary.

# Feedback Guidelines
# Provide specific and impactful feedback based on your interaction:

# Content and Usability: Comment on the page's layout, information presentation, and ease of use.
# Design and Functionality: Note any design aspects or functionalities that enhance or hinder the experience.
# Personal Relevance: Reflect on how the website meets your persona's needs and expectations.
# Improvement Suggestions: Offer constructive suggestions for improvement.
# Focus on how well the website is being able to facilitate how well you can perform the tasks you want to.
# Example Feedback:
# The product descriptions are clear and give me a general idea of what to expect. However it would be nice to have some sort of reviews also displayed so I can understand pros and cons of each product beforehand. It would help me pick better products and facillitate better shopping."""
        
    def vision_test(self,api_key,img,img2, img3 ,flag2,flag3,website_context,past_context, clickable_elements,user_contect,searchable_elements,feedback):
        observations=self.memory.fetch_memories(website_context)
        observation_str = "\n".join(
            [self.memory._format_memory_detail(o) for o in observations]
        )


        summary=str(self.get_summary())
        client = OpenAI(
        api_key=api_key)
        headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
               "type": "text",
                 "text":  f"""You are an AI agent testing a website as a specific user persona. Your task is to interact with the website and provide detailed feedback. You will be returning a dict in the end so keep this in mind.

Website Context
Context: {website_context}
\n
Your Persona
Here is a Summary of yourself: {summary}
\n
Task Information: 
Here is the specific task you said you wanted to complete on this website: {user_contect}. Use this information to guide what you will do on the website. Your goal is to complete this task as best as possible. So efficiently navigate the website and choose what to do based on what will help you most efficiently reach your goal and achieve the results you want. Try to maximize your operations so that you are achieving your task in the best way possible.

Task Instructions:
You are going to either choose if you want to click an element or search something up or scroll down, based on the elments in the screenshot.
Interaction:
Clickable Elements: Here is a list of all clickable elements: {clickable_elements}, If you want to click something choose something from here. Be aware there may be some elements here that are not visible. Only pick what is visible. DO not pick an element that is not visible on the screenshot. This will not work. Only click what you can see. If you see something on clickable elements but it is not in the screenshot do not pick it.
Searchable Elements: Here is a list of all the searchable elements placeholder texts: {searchable_elements}. Some search elements might not have placeholder text so if this list is empty but you clearly see a searchable elements still return what you want to search with empty placeholder text. Try to choose a placeholder text if you are searching something up though.
Action Format:
For clicks, you should this key value pair to the dict you will eventually return button: [item name]. Example: button: Submit
For typing, you should add this key value pair to the dict you will eventurally return return search: [placeholder text: search value]. Example: search: placeholder text: 240 dollars
For scrolling down you will eventually return scroll: down. Example: scroll[down]
Note: Only interact with elements visible in the screenshot. Be careful when searching values up. Sometimes you have to click in the element before searching something up. If there is a searchbar with placeholder text check if it is in the clickable elements first before returning search. If it is in clickable elements and you have not clicked it then click it before searching. Be aware that the placeholder text is the text in the searchbar. 
For scrolling be aware you might want to scroll if there is information that you need that may be located more down. Do not be afraid to scroll if some information you want is cut off or if you believe there is more relevant information if you scroll.
You are trying to emulate a person so make your actions similair to how an actual person would navigate a website.

Past Interactions:
Avoid repeating past actions(except scrolling). Here is a list of what you have previosly clicked along with reasoning for why you clicked each value: {past_context} for history.

Focus on unique elements unless repetition is necessary. Use the past_context to inform your next decision. If you saw the previously clicked something on the website build on that to complete your task. Use the past_context as a guide to inform your next decision. They should be building blocks to your final goal. DO not repeat actions excessively as that will prevent you from finishing the task.

Warnings:
Be aware of popups. If you see a popup you will probably need to hit the button to exit or get out of the popup. Click the necessary element to keep progressing through the website. If you do not you will just be stuck on the popup.
Popups:
If you see a popup blocking content hit the neccesary button to continue from the popup. Interact with the popup as it will be the only way to continue. You also cannot search things up when there is a popup. Keep this in mind. Popups black activity.

Feedback Guidelines
Provide specific and impactful feedback based on your interaction:

Content and Usability: Comment on the page's layout, information presentation, and ease of use.
Design and Functionality: Note any design aspects or functionalities that enhance or hinder the experience.
Personal Relevance: Reflect on how the website meets your persona's needs and expectations.
Improvement Suggestions: Offer constructive suggestions for improvement.
Focus on how well the website is being able to facilitate how well you can perform the tasks you want to. Only provide feedback on what you can see. Be careful when providing feedback. Talk about what you might be confused by and what you like and be aware that you may have not interacted with certain features yet. 
Do not overdo feedback. Do not say your are confused by something if you are not. Provide accurate feedback that will be helpful for the company. Only provide relevant feedback that will prodivde some value and directly relates to what you are interacting with.

Example Feedback:
The product descriptions are clear and give me a general idea of what to expect. However it would be nice to have some sort of reviews also displayed so I can understand pros and cons of each product beforehand. It would help me pick better products and facillitate better shopping.

Reasoning: 
Provide reasoning for why you are clicking/searching/scrolling. Focus on why you believe it will advance the task you want to complete on the website.


Return Value: 
Here is what you should return. You should return a dict with what you want to click/searh/scroll, feedback, and reasoning. The keys should be either button/search/scroll, feedback, and reasoning. Make sure the keys are exaclt this and spelled like this.
Example return:
button: Submit, feedback: your feedback here, reasoning: your reasoning here.
These are the keys and values you should be returning the dictionary you are returning and only this. Do not add any extra keys to the anwser.
Only return a dictionary and nothing else other than they keys mentioned above. Return only a dictionary. The dictionary should have the keys button/search/scroll, feedback, and reasoning. Make sure you are return a proper python dictionary. Use quotations in this dictionary
"""
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }
        payload1={  "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                  "type": "text",
           "text":   f"""As an AI agent simulating a user experience on a website, your task is to interact with the website and provide detailed feedback. Return a dictionary containing your actions, feedback, and reasoning.

Website Context:
- Context: {website_context}
- Images: You'll receive two images - the website's UI before and after your last action. Analyze the changes to guide your next action. If the images are identical, your last action was ineffective, so do not repeat it.

Your Persona:
- Summary: {summary}

Task Information:
- Objective: {user_contect}. Your goal is to efficiently navigate the website to complete this task. Choose your actions so you are efficiently completing this task.

Interaction Options:
- Clickable Elements: {clickable_elements}. Choose visible elements only.
- Searchable Elements: {searchable_elements}. If an element has no placeholder text but is clearly searchable, you can still use it.
- Scrolling: Scroll down if necessary to uncover more information or complete the task.

Action Format:
- Click: 'button': '[item_name]' (e.g., 'button': 'Submit')
- Search: 'search': '[placeholder_text: search_value]' (e.g., 'search': 'placeholder: 240 dollars')
- Scroll: 'scroll': 'down' (e.g., 'scroll': 'down')
Note: For search bars, click them first if they're listed in clickable elements.

Past Interactions:
- History: {past_context}. Use this to avoid repetitive actions and build on previous steps.

Warnings:
- Popups: Interact with popups promptly to continue navigating the site.

Feedback Guidelines:
- Content and Usability: Assess layout and information presentation.
- Design and Functionality: Note design elements and functionality.
- Personal Relevance: Reflect on how the site meets your persona's needs.
- Improvement Suggestions: Offer constructive ideas.
- Past Feedback: {feedback}. Provide unique feedback; avoid repetition.

Reasoning:
- Justify your actions based on task advancement.

Return Value:
- Format: {'button/search/scroll': 'action', 'feedback': 'your feedback', 'reasoning': 'your reasoning'}.
- Ensure accuracy and relevance in your feedback and actions.

Example Return:
{'button': 'Submit', 'feedback': 'The product descriptions are clear...', 'reasoning': 'This will help me...'}
"""
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}"
                }
                },
                 {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img2}"
                }
                }

            ]
            }
        ],
        "max_tokens": 300,
        "temperature": .9
        }
        payload2={  "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
           {
                "type": "text",
                  "text":     f"""You are an AI agent testing a website as a specific user persona. Your task is to interact with the website and provide detailed feedback. You will be returning a dict in the end so keep this in mind.

Website Context
Context: {website_context}
\n
Your Persona
Here is a Summary of yourself: {summary}
\n
Task Information: 
Here is the specific task you said you wanted to complete on this website: {user_contect}. Use this information to guide what you will do on the website. Your goal is to complete this task as best as possible. So efficiently navigate the website and choose what to do based on what will help you most efficiently reach your goal and achieve the results you want. Try to maximize your operations so that you are achieving your task in the best way possible.

Task Instructions:
You are going to either choose if you want to click an element or search something up or scroll down, based on the elments in the screenshot.
Interaction:
Clickable Elements: Here is a list of all clickable elements: {clickable_elements}, If you want to click something choose something from here. Be aware there may be some elements here that are not visible. Only pick what is visible. DO not pick an element that is not visible on the screenshot. This will not work. Only click what you can see. If you see something on clickable elements but it is not in the screenshot do not pick it.
Searchable Elements: Here is a list of all the searchable elements placeholder texts: {searchable_elements}. Some search elements might not have placeholder text so if this list is empty but you clearly see a searchable elements still return what you want to search with empty placeholder text. Try to choose a placeholder text if you are searching something up though.
Action Format:
For clicks, you should this key value pair to the dict you will eventually return button: [item name]. Example: button: Submit
For typing, you should add this key value pair to the dict you will eventurally return return search: [placeholder text: search value]. Example: search: placeholder text: 240 dollars
For scrolling down you will eventually return scroll: down. Example: scroll[down]
Note: Only interact with elements visible in the screenshot. Be careful when searching values up. Sometimes you have to click in the element before searching something up. If there is a searchbar with placeholder text check if it is in the clickable elements first before returning search. If it is in clickable elements and you have not clicked it then click it before searching. Be aware that the placeholder text is the text in the searchbar. 
For scrolling be aware you might want to scroll if there is information that you need that may be located more down. Do not be afraid to scroll if some information you want is cut off or if you believe there is more relevant information if you scroll.
You are trying to emulate a person so make your actions similair to how an actual person would navigate a website.

Past Interactions:
Avoid repeating past actions(except scrolling). Here is a list of what you have previosly clicked along with reasoning for why you clicked each value: {past_context} for history.

Focus on unique elements unless repetition is necessary. Use the past_context to inform your next decision. If you saw the previously clicked something on the website build on that to complete your task. Use the past_context as a guide to inform your next decision. They should be building blocks to your final goal. DO not repeat actions excessively as that will prevent you from finishing the task.

Warnings:
Be aware of popups. If you see a popup you will probably need to hit the button to exit or get out of the popup. Click the necessary element to keep progressing through the website. If you do not you will just be stuck on the popup.
Popups:
If you see a popup blocking content hit the neccesary button to continue from the popup. Interact with the popup as it will be the only way to continue. You also cannot search things up when there is a popup. Keep this in mind. Popups black activity.

Feedback Guidelines
Provide specific and impactful feedback based on your interaction:

Content and Usability: Comment on the page's layout, information presentation, and ease of use.
Design and Functionality: Note any design aspects or functionalities that enhance or hinder the experience.
Personal Relevance: Reflect on how the website meets your persona's needs and expectations.
Improvement Suggestions: Offer constructive suggestions for improvement.
Focus on how well the website is being able to facilitate how well you can perform the tasks you want to. Only provide feedback on what you can see. Be careful when providing feedback. Talk about what you might be confused by and what you like and be aware that you may have not interacted with certain features yet. 
Do not overdo feedback. Do not say your are confused by something if you are not. Provide accurate feedback that will be helpful for the company. Only provide relevant feedback that will prodivde some value and directly relates to what you are interacting with.
Example Feedback:
The product descriptions are clear and give me a general idea of what to expect. However it would be nice to have some sort of reviews also displayed so I can understand pros and cons of each product beforehand. It would help me pick better products and facillitate better shopping.

Reasoning: 
Provide reasoning for why you are clicking/searching/scrolling. Focus on why you believe it will advance the task you want to complete on the website.


Return Value: 
Here is what you should return. You should return a dict with what you want to click/searh/scroll, feedback, and reasoning. The keys should be either button/search/scroll, feedback, and reasoning. Make sure the keys are exaclt this and spelled like this.
Example return:
button: Submit, feedback: your feedback here, reasoning: your reasoning here.
These are the keys and values you should be returning the dictionary you are returning and only this. Do not add any extra keys to the anwser.
Only return a dictionary and nothing else other than they keys mentioned above. Return only a dictionary. The dictionary should have the keys button/search/scroll, feedback, and reasoning. Make sure you are return a proper python dictionary. Use quotations in this dictionary
"""
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}"
                }
                },
                 {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img2}"
                }
                },
                    {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img3}"
                }
                },

            ]
            }
        ],
        "max_tokens": 300,
        "temperature": .5
        }
        if flag2 and not flag3: 
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload1)

            return(response.json())
        elif flag3: 
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload2)

            return(response.json())
        else: 
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            return(response.json())
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

    def __getstate__(self):
        # Return everything except the lock
        state = self.__dict__.copy()
        del state['lock']
        return state

    def __setstate__(self, state):
        # Restore the object's state and reinitialize the lock
        self.__dict__.update(state)
        self.lock = threading.Lock()