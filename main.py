import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.prebuilt import create_react_agent
from database import DatabaseManager


load_dotenv()

api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API KEY is missing in env")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE API KEY is missing in env")

app = FastAPI()
sessions = {}
db_manager = DatabaseManager("restaurant.db")

class ChatMessage(BaseModel):
    session_id: str
    message: str

    @field_validator("session_id")
    @classmethod
    def session_id_is_missing(cls, v):
        if not v.strip():
            raise ValueError("Session ID is missing")
        return v
    
    @field_validator("message")
    @classmethod
    def message_is_empty(cls, v):
        if not v.strip():
            raise ValueError("Message is empty")
        return v
    
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


pc = Pinecone(api_key=pinecone_api_key)

if "bella-italia-v2" not in pc.list_indexes().names():
    pc.create_index(
        name="bella-italia-v2",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

def build_pipeline():
    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    menu_loader = PyPDFLoader("menu.pdf")
    menu_docs = menu_loader.load()
    print(f"Menu pages loaded: {len(menu_docs)}")
    menu_chunks = splitter.split_documents(menu_docs)
    all_chunks.extend(menu_chunks)

    faq_loader = TextLoader("faq.txt")
    faq_docs = faq_loader.load()
    print(f"FAQ documents loaded: {len(faq_docs)}")
    faq_chunks = splitter.split_documents(faq_docs)

    all_chunks.extend(faq_chunks)
    print(f"Total chunks: {len(all_chunks)}")

    return all_chunks
    

vector_store = PineconeVectorStore(
    index_name="bella-italia-v2",
    embedding=embeddings,
    pinecone_api_key=pinecone_api_key
)

index = pc.Index("bella-italia-v2")
stats = index.describe_index_stats()

if stats.total_vector_count == 0:
    chunks = build_pipeline()
    vector_store.add_documents(chunks)
    print("Documents loaded successfully")
else:
    print("Documents already loaded - skipping")

llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=500,
        api_key=api_key
    )

@tool
def search_menu_rag(query: str) -> str:
    """Searches the restaurant menu using semantic search.
    Use this for any question about food, drinks or prices."""
    results = vector_store.similarity_search(query, k=3)
    context = ""
    for doc in results:
        context += f"{doc.page_content}\n\n"
    return context if context else "No menu information found."

@tool
def search_faq_rag(query: str) -> str:
    """Searches the restaurant FAQ using semantic search.
    Use this for questions about hours, parking, policies etc."""
    results = vector_store.similarity_search(query, k=3)
    context = ""
    for doc in results:
        context += f"{doc.page_content}\n\n"
    return context if context else "No FAQ information found."

@tool
def check_dietary_options(requirement: str) -> str:
    """It checks whether specific deitary option is available or not.
    Use this for any dietary related questions"""

    dietary_options = ["vegetarian", "vegan", "gluten_free"]

    requirement = requirement.lower()

    if requirement in dietary_options:
        return f"Yes, we have available {requirement} option"
    
    return f"No, we don't have {requirement} option available."

@tool
def check_table_availability(date: str, time: str, people: str) -> dict:
    """Checks if a table is available for a specific date, time and number of people.
    Use this BEFORE booking to verify availability.
    Date format must be YYYY-MM-DD like 2026-05-04.
    Time can be any format like 7 PM, 19:00 or 7:00 PM.
    People must be a number like 4."""
    people = int(people)
    return db_manager.check_availability(date, time, people)

@tool
def book_table(customer_name: str, date: str, time: str , people: str, special_requirement: str = None) -> dict:
    """Books a table at the restaurant for a customer.
    Use this AFTER checking availability.
    Requires customer name, date in YYYY-MM-DD format, time and number of people.
    Special requirement is optional — use for dietary needs or special occasions.
    Always check availability first before calling this tool."""
    people = int(people)
    return db_manager.book_with_validation(customer_name, date, time, people, None, None, special_requirement)

@tool
def get_my_reservation(reference: str) -> dict:
    """Retrieves reservation details by reference number.
    Use this when customer asks about their existing booking.
    Requires the reference number given at time of booking."""
    reference = int(reference)
    return db_manager.get_reservation_by_reference(reference)

@tool
def find_reservations_by_name(name: str) -> dict:
    """Finds all reservations for a customer by their name.
    Use this when customer provides their name and wants to see their bookings.
    Returns all confirmed reservations sorted by date and time."""
    return db_manager.get_reservations_by_name(name)

@tool
def cancel_my_reservation(reference: str) -> dict:
    """Cancels an existing reservation by reference number.
    Use this when customer wants to cancel their booking.
    Requires the reference number.
    Always confirm with customer before cancelling."""
    reference = int(reference)
    return db_manager.cancel_reservation(reference)

@tool
def update_my_reservation(reference: str, new_date: str = None, new_time: str= None, new_people: str= None ) -> dict:
    """Updates an existing reservation by reference number.
    Use this when customer wants to change their booking details.
    Only provide the fields that need to be changed.
    Reference number is required.
    New date format must be YYYY-MM-DD.
    New time can be any format like 7 PM or 19:00."""
    reference = int(reference)
    new_people = int(new_people) if new_people else None
    return db_manager.update_reservation(reference, new_date, new_time, new_people)

@tool
def ask_menu_agent(question: str) -> str:
    return None

config = db_manager.get_config()

@tool
def get_restaurant_info():
    """Returns restaurants Information
    Use this for restaurant information"""
    return f"Name: Bella Italia\nOpening Hours: {config['opening_time']} to {config['closing_time']}\nLocation: Astoria, New York\nPhone: 123-456-7890"


menu_tools = [search_menu_rag, check_dietary_options]

reservation_tools = [
    check_table_availability,
    book_table,
    get_my_reservation,
    find_reservations_by_name,
    cancel_my_reservation,
    update_my_reservation,
    ask_menu_agent
]

faq_tools = [search_faq_rag, get_restaurant_info]

menu_prompt = """You are Marco, menu specialist for Bella Italia.
    Only handle food, drinks, dietary and price questions.
    Always use search_menu_rag() for menu questions.
    Always use check_dietary_options() for dietary questions.
    Never handle bookings or general questions.
    Never make up menu items."""

reservation_prompt = """You are Sofia, reservation specialist for Bella Italia.
    Only handle bookings, cancellations, updates and lookups.
    Always check availability before booking.
    Always gather name, date, time and people before booking.
    If customer mentions dietary requirement use ask_menu_agent() first.
    Date format YYYY-MM-DD.
    Time any format — system handles conversion."""

faq_prompt = """You are Luca, customer service specialist for Bella Italia.
    Handle general questions about the restaurant.
    Always use search_faq_rag() for policy questions.
    Always use get_restaurant_info() for basic info.
    Never handle menu or booking questions."""


menu_agent = create_react_agent(llm, menu_tools, prompt=menu_prompt)
reservation_agent = create_react_agent(llm, reservation_tools, prompt=reservation_prompt)
faq_agent = create_react_agent(llm, faq_tools, prompt=faq_prompt)

agents = {
    "menu": menu_agent,
    "reservation": reservation_agent,
    "faq": faq_agent
}

def get_session(session_id: str) -> list:
    if session_id not in sessions:
        sessions[session_id] = []

    return sessions[session_id]

def route_message(message: str, history: list) -> str:
    """Routes message to correct agent using history context."""

    history_text = ""
    if history:
        recent = history[-4:]
        for msg in recent:
            role = msg["role"].upper()
            history_text += f"{role}: {msg['content']}\n"

    response = llm.invoke([
        HumanMessage(content=f"""
        You are a router for a restaurant AI system.
        Classify which agent should handle this message.

        Agents available:
        - menu: questions about food, drinks, dietary options, prices
        - reservation: booking tables, cancellations, updates, existing bookings
        - faq: opening hours, location, parking, payment, policies, general info

        Conversation history:
        {history_text}

        Current message: {message}

        Examples:
        "Do you have vegan pizza?" → menu
        "Book a table for 4" → reservation
        "Cancel my booking" → reservation
        "What time do you open?" → faq
        "Do you have parking?" → faq
        "Book one for me" → reservation (follow up to booking intent)
        "How much does it cost?" → menu (follow up to menu question)

        Reply with only one word: menu, reservation, or faq
        """)
    ])

    route = response.content.strip().lower()

    if route not in ["menu", "reservation", "faq"]:
        return "faq"

    return route

@app.post("/chat")
def chat(message: ChatMessage):
    session_id = message.session_id
    query = message.message

    history = get_session(session_id)

    route = route_message(query, history)
    history.append(HumanMessage(content=query))

    result = agents[route].invoke({"messages": history})
    ai_message = result["messages"][-1]

    history.append(ai_message)

    return {
        "answer": ai_message.content,
        "routed_to": route,
        "session_id": session_id,
        "history_length": len(history)
    }

