import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from typing import Optional
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec