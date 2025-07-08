import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

search_tool = SerperDevTool()

def create_agent():

    llm = ChatOpenAI(model="gpt-3.5-turbo")
  
    return Agent(
        role="Research Specialist",
        goal="Conduct strong research on given topics",
        backstory="You are an expert researcher with strong knowledge in finding and synthesizing information from various sources",
        verbose=True,  # Result will be displayed in terminal
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
    )




def task_agent(agent, topic):
    return Task(
        description=f"Research the topic and provide a comprehensive summary: {topic}",
        agent=agent,
        expected_output = "A detailed summary of the research findings, including key points and insights related to the topic"
    )

def run_crew(topic):
    agent = create_agent()
    task = task_agent(agent, topic)
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()   # Start the crew (workflow)
    return result

if __name__ == "__main__":
    print("Welcome to the Research Agent!")
    topic = input("Enter the topic you want to know about: ")
    result = run_crew(topic)
    print("Result:")
    print(result)