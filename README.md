# Bella Italia — Multi-Agent AI Backend

A complete restaurant AI system combining multi-agent architecture,
RAG document search and a real database with full validation.
Three specialized agents handle menu questions, table reservations
and FAQ queries — with a context aware router and agent communication.

## Features

- Three specialized agents — Marco (menu), Sofia (reservations), Luca (FAQ)
- Context aware routing — uses conversation history for better decisions
- Agent communication — reservation agent consults menu agent for dietary needs
- RAG search — agents answer from real PDF and TXT documents
- Real database — SQLite with full CRUD and validation
- Time validation — rejects bookings outside opening hours
- Capacity management — prevents overbooking
- Time normalization — accepts any time format like 7 PM or 19:00
- Soft delete — cancellations preserved as history
- Pinecone vector database — embeddings stored permanently in cloud
- Session memory — each customer has separate conversation history

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | Backend web framework |
| LangGraph | Multi-agent framework |
| LangChain | AI tooling and RAG |
| Pinecone | Cloud vector database |
| HuggingFace | Free local embedding model |
| Groq API | AI language model |
| LLaMA 3.3 70B | AI model |
| SQLite | Reservation database |
| PyPDF | PDF document reading |
| Pydantic | Data validation |
| python-dotenv | Environment variable management |

## Project Structure
```
multi-agent-ai-backend/
│
├── env/
├── main.py
├── database.py
├── menu.pdf
├── faq.txt
├── .env
└── requirements.txt
```
## Setup

1. Clone the repository
```
git clone https://github.com/yourusername/bella-italia-multi-agent-ai-backend
```
2. Create and activate virtual environment
```
python -m venv env
env\Scripts\activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Create `.env` file
```
API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```
5. Add documents to project folder
```
menu.pdf  →  restaurant menu PDF
faq.txt   →  frequently asked questions
```
6. Run the server
```
uvicorn main:app --reload
```
## API Endpoints

### POST /chat
Main AI chatbot endpoint with multi-agent routing.

**Request:**
```json
{
    "session_id": "user_1",
    "message": "Do you have vegan options?"
}
```

**Response:**
```json
{
    "answer": "Yes we have Vegan Arrabbiata — spicy tomato pasta with no animal products at $12.",
    "routed_to": "menu",
    "session_id": "user_1",
    "history_length": 2
}
```

### GET /reservations
```
Returns all confirmed reservations.
GET /reservations
GET /reservations?date=2026-12-25
```
## Agent System

### Menu Agent — Marco
Handles all food and drink related questions.

| Tools | Purpose |
|---|---|
| `search_menu_rag` | Searches menu PDF using semantic search |
| `check_dietary_options` | Checks dietary availability |

### Reservation Agent — Sofia
Handles all booking related requests.

| Tools | Purpose |
|---|---|
| `check_table_availability` | Checks availability with validation |
| `book_table` | Books with full time and capacity validation |
| `get_my_reservation` | Retrieves booking by reference |
| `find_reservations_by_name` | Finds all bookings by name |
| `cancel_my_reservation` | Soft cancels booking |
| `update_my_reservation` | Updates booking details |
| `ask_menu_agent` | Consults menu agent for dietary info |

### FAQ Agent — Luca
Handles general restaurant questions.

| Tools | Purpose |
|---|---|
| `search_faq_rag` | Searches FAQ document using semantic search |
| `get_restaurant_info` | Returns restaurant details |

## Routing System
```
Customer message
↓
Context aware router — uses last 4 messages of history
↓
menu        →  Marco handles food and drink questions
reservation →  Sofia handles bookings and cancellations
faq         →  Luca handles general questions
↓
Response returned with routed_to field
```
## Agent Communication
```
Customer: "I am vegan — book a table for 4"
↓
Router → reservation agent
↓
Sofia detects dietary requirement
↓
Sofia calls ask_menu_agent() → consults Marco
↓
Marco confirms vegan options available
↓
Sofia books table with vegan special requirement
```
## Database Schema

```sql
CREATE TABLE reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT NOT NULL,
    customer_phone TEXT,
    customer_email TEXT,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    people INTEGER NOT NULL,
    special_requirement TEXT,
    status TEXT DEFAULT 'confirmed',
    reference INTEGER UNIQUE NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)

CREATE TABLE restaurant_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    max_capacity INTEGER DEFAULT 50,
    opening_time TEXT DEFAULT '12:00',
    closing_time TEXT DEFAULT '23:00',
    slot_duration INTEGER DEFAULT 90
)
```

## Validation Rules
```
Time: must be within opening hours
Date: must be in the future
Capacity: maximum 50 people per slot
Time formats: 7 PM, 19:00, 7:00 PM, 7pm all accepted
```
## Document Pipeline
```
menu.pdf + faq.txt
↓
Loaded and split into chunks
↓
Embedded with HuggingFace all-MiniLM-L6-v2
↓
Stored permanently in Pinecone
↓
Agents search by meaning not keywords
```

## Environment Variables
```
API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```
## Notes

- Never commit your .env file to GitHub
- Documents embedded once — skipped on subsequent restarts
- Pinecone free tier — 1 index, 2GB, no credit card
- Session memory resets when server restarts
- routed_to field shows which agent handled each 

## 👤 Author

**Ohm Parkash** — [LinkedIn](https://www.linkedin.com/in/om-parkash34/) · [GitHub](https://github.com/parkash34)