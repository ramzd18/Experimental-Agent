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
            "Summary of {name}: "
            "{summary}"
            " Here ia a description of the person that was inputted: {description}"
            "This description likely details a problem or general persona a person fits under."
            "Given theis relevant information from a persons memories, what are twelve relevant things you think they would search up to fit the desciprtion that was inputted. \n"
            "Tailor the questions so you are covering what products this persons may use to solve their problems/ fit their persona.For example if the person inputted description of a person struggling to mantain personal finances, you might look up different budgeting products online, their pros/and cons, the basics of maintaining budget, and other relevant things to help you be more knowledagble about the topic."
            " Make sure the questions relate to {description}."
            "Seperate each thing you want to learn with ;."
        )
        summary=self.get_summary()
        result = self.chain(prompt).run(name=self.name,summary=summary,description=description)
        result= result.split(";")
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
            +"Make sure the response is unique to you and something you believe you would say. Do not just give the most common anwser but anwser what makes sense to you and your profile."
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
        result= self.chain1(prompt).run(**kwargs).strip()
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
    def memoriesprompt(self,mems,now: Optional[datetime]=None):
        prompt = PromptTemplate.from_template(
            "I am trying to rebuild a persons memory. I have soem of their memories and information and I want to rebuild it with the information. Use the following information to generate additional memories for this person.I do not want sufrace level memories but rather deep, detailed memories core to the person."
            "{agent_summary_description}"
            # + "\n{agent_name}'s status: {agent_status}"
            +"\n {agent_name}'s interests:"
            + "\nAll of  {agent_name}'s current memories:"
            + "\n{relevant_memories}"
            + "\n\n"
            +"\n Use the following information to generate a comprehensive list of additional memories the person may have to bolster the information we have on them. The memories {agent_name} already contains are limited and  I want you to produce more memories to make the information about {agent_name} more complete. Generate the memories with extensive detail. Seperate each memory with a semicolon.Example format: mem1;mem2;mem3;mem4 ..."
            # + suffix
        )
    
        # agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = mems
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=self.education_and_work+ "    "+ self.interests,
            relevant_memories=relevant_memories_str,
            agent_name= self.name,
        )
        result= self.chain(prompt).run(**kwargs)
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
    def analysis_of_product(self,list_of_text):
        total_len= len(list_of_text)
        iter= total_len//10
        if (total_len<9): 
            iter=1
        begin=0
        end=iter
        totallist=[]
        while(end<total_len-1): 
            sublist= list_of_text[begin:end]
            resultque=queue.Queue()
            task_thread = threading.Thread(target=self.memoryfunc, args=(sublist,resultque))
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

    def memoryfunc(self,list,resultque): 
            prompt = PromptTemplate.from_template(
            "Here is a list of summarized articles {name} searched up on the internet. "
            "{observation_str}\n"
            "---\n"
            "Here are a summary of {name}'s relevent social media interactions towards the topics of these articles:"
            "{social_str}\n"
            "---\n"
            "Here is a summary of {name}: {summary}"
            "Here are {name}'s interests: {interests} \n"
            #  "{name}'s current status: {status} \n"
            " Imagine that you are {name}. Given this generate a list of memories {name} would remember based on reading these articles. Write the memories from the perspective of {name}. Make sure they are personalized memories. "
            "Write as many memories as you can. Seperate the memories with a semicolon."
            "For example, if the person we were reading an article about basketball shoes, and the person enjoyed playing basketball a memory coud be I played countless games of pick up basketball with my friends and tried jumping so hard my shoe broke."
            "Avoid using works like I remember or I recall or I am feeling. and instead state the memory directly and include extremely specific details in the memory so they are not broad or general. Be as creative as you can be. Let the memories be unique and use the articles and the persons profile to make the most realistic human like memories possible. Do not just reuse the information from the articles. Think about how they might have applied to you and your life and make sure you are unqiue to the person."
            "Here is an example format memory1;memory2;memory3;memory4;memory5;memory6 and so on"
        )
            soc_mem=self.summarize_related_memories(str(list))
            result =self.chain1(prompt).run(observation_str=str(list),name=self.name,social_str=soc_mem,summary=self.get_summary(),interests=str(self.interests),status=self.status)
            result=result.split(";")
            print(result) 
            resultque.put(result)
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
        
    def vision_test(self,api_key,img,img2, img3 ,flag2,flag3,website_context,past_context, clickable_elements):
        observations=self.memory.fetch_memories(website_context)
        observation_str = "\n".join(
            [self.memory._format_memory_detail(o) for o in observations]
        )
        print("Length of observations"+str(len(observation_str)))
        print("Length of observations"+str(len(self.get_summary())))

        summary=str(self.get_summary())+ "Here are relevant memories you have related to the topic."+ str(observation_str)
        print(len(summary))
        print(summary)
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
                "text": f"Here is context of the website you are looking at.{website_context}.Here is a summary of yourself: {summary}."
                 +" Given this picture of the website respond with wheter you want to click something on the website or if you want to type something into an element. Respond with your anwser in a dictionary with either button or search as they key and object name you are clicking or search value as the value. If you want to click on image or text that you think is clickable provide the text or image caption as the clickable value in the dictionary."
                 +"For example if you wanted to click a button named Submit you would return button: Submit. If you wanted to click a text labeled Click This you would return button: Click This. If you wanted to type 240 dollars into a box you would return search: 240 dollars. Use quotations. Only return this value and nothing else."
                 +f"Sometimes it may be hard for you to identify if something is clickable. Here is a list of the clickable elements you can choose on the page to help you determine what is clickable: {clickable_elements}. You may not see some of these elements in the screensot. If you want to choose an element in the list and is not in the picture do not pick it. Only pick what you can see."
                +f" Here is the past things you have searched/click on the website: {past_context}. If you  something was already clicked or search in the past thigns you have searched/click do not return it again. DO NOT REPEAT things. For example if you clicked a buttonn called Best Clothes Here, do not pick that button again if you have choose it recently. This will cause issues by causing you to go in circles."
                  +" Make it realistic to how your profile might use the website."
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
                "text": f"Here is context of the website you are looking at.{website_context}.Here is a summary of yourself: {summary}."
                 +" Given these sets of pictures of the website respond with wheter you want to click something on the website or if you want to type something into an element. Respond with your anwser in a dictionary with either button or search as they key and object name you are clicking or search value as the value. If you want to click on image or text that you think is clickable provide the text or image caption as the clickable value in the dictionary."
                 +"For example if you wanted to click a button named Submit you would return button: Submit. If you wanted to click a text labeled Click This you would return button: Click This. If you wanted to type 240 dollars into a box you would return search: 240 dollars. Use quotations. Only return this value and nothing else."
                 +f"Sometimes it may be hard for you to identify if something is clickable. Here is a list of the clickable elements you can choose on the page to help you determine what is clickable: {clickable_elements}. You may not see some of these elements in the screensot. If you want to choose an element in the list and is not in the picture do not pick it. Only pick what you can see."
                +f" Here is the past things you have searched/click on the website: {past_context}. If you  something was already clicked or search in the past thigns you have searched/click do not return it again. DO NOT REPEAT things. For example if you clicked a buttonn called Best Clothes Here, do not pick that button again if you have choose it recently. This will cause issues by causing you to go in circles."
                  +" Make it realistic to how your profile might use the website."
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
        "max_tokens": 300
        }
        payload2={  "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"Here is context of the website you are looking at.{website_context}.Here is a summary of yourself: {summary}."
                 +" Given these sets of pictures of the website respond with wheter you want to click something on the website or if you want to type something into an element. Respond with your anwser in a dictionary with either button or search as they key and object name you are clicking or search value as the value. If you want to click on image or text that you think is clickable provide the text or image caption as the clickable value in the dictionary."
                 +"For example if you wanted to click a button named Submit you would return button: Submit. If you wanted to click a text labeled Click This you would return button: Click This. If you wanted to type 240 dollars into a box you would return search: 240 dollars. Use quotations. Only return this value and nothing else."
                 +f"Sometimes it may be hard for you to identify if something is clickable. Here is a list of the clickable elements you can choose on the page to help you determine what is clickable: {clickable_elements}. You may not see some of these elements in the screensot. If you want to choose an element in the list and is not in the picture do not pick it. Only pick what you can see."
                +f" Here is the past things you have searched/click on the website: {past_context}. If you  something was already clicked or search in the past thigns you have searched/click do not return it again. DO NOT REPEAT things. For example if you clicked a buttonn called Best Clothes Here, do not pick that button again if you have choose it recently. This will cause issues by causing you to go in circles."
                  +" Make it realistic to how your profile might use the website."
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
        "max_tokens": 300
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